#!/usr/bin/env python3
#===============================================================================
# Copyright Contributors to the oneDAL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

"""Linkage half of the Make-vs-Bazel release comparison.

`compare_release_trees.py` answers "do the two releases contain the same files,
with the same contents and the same exported symbols". This module answers the
other question a release has to pass: "can what it contains actually be loaded".
That means each shared library's DT_NEEDED list, the dynamic symbols it leaves
for someone else to define, and the third-party runtimes the package has to ship
next to the oneDAL libraries for those dependencies to resolve.

Kept in its own module because the comparison script is already well past the
repository's file-size guidance, and because these checks read ELF metadata with
external tools -- a self-contained concern with its own ignore rules.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path


SHARED_LIBRARY_SUFFIXES = {
    "linux": (".so",),
    "windows": (".dll",),
}


# Platforms whose shared libraries record their dependencies and unresolved
# references in a form this script knows how to read. Windows DLLs do carry an
# import table, but reading it needs `dumpbin /DEPENDENTS` and `/IMPORTS`, whose
# output this script does not parse yet, so the linkage comparison announces
# itself as skipped there rather than silently reporting nothing.
DYNAMIC_LINKAGE_PLATFORMS = ("linux",)

# DT_NEEDED entries that may legitimately differ between the two builds. Kept
# empty on purpose: a dependency a released library records is part of what a
# consumer's loader has to satisfy, so every difference is reported until one is
# shown to be a build-system detail rather than a linkage one.
LINUX_IGNORED_NEEDED = set()

# Undefined dynamic references get their own ignore rules rather than reusing the
# export ones. The export list drops everything under `_ZN3tbb`, because which
# oneTBB template instantiation a build happens to emit as a *definition* is not
# part of the oneDAL surface -- but an undefined oneTBB *reference* is precisely
# the signal this comparison exists for: it is what proves a missing
# `DT_NEEDED libtbb.so.12` is a real defect rather than a dependency the library
# does not need. Reusing the export list would hide all 30 `tbb::` references in
# libonedal_thread and all 20 in libonedal_core.
#
# What is genuinely not a linkage property is how the compiler and the C++
# standard library implementation choose to emit their own internals: which
# `std::` instantiation gets inlined, which vtable or type_info lands in which
# object, and whether exception and guard helpers are referenced directly.
# Measured between a Make and a Bazel release built from the same commit with the
# same gcc: of the 1261 undefined references the four Linux libraries carry, 5
# differ. These prefixes cover 4 of them (the fifth is named below), and the 1015
# references that survive the filter -- including all 64 `tbb::` ones -- match
# entry for entry.
LINUX_IGNORED_UNDEFINED_PREFIXES = (
    # libstdc++ templates, their vtables and type_info, and the stream classes
    # (`std::basic_ostream` members mangle as `_ZNSo...`, not `_ZSo...`).
    "_ZSt",
    "_ZNSt",
    "_ZNKSt",
    "_ZTVSt",
    "_ZTISt",
    # The `__cxx11` classes nest one level deeper, so their vtables and type_info
    # mangle as `_ZTVNSt7__cxx1119basic_ostringstream...` -- `_ZTVSt` misses them.
    "_ZTVNSt",
    "_ZTINSt",
    "_ZNSi",
    "_ZNKSi",
    "_ZNSo",
    "_ZNKSo",
    "_ZNSb",
    "_ZNKSb",
    # `operator new` / `operator delete` in all their spellings.
    "_Znw",
    "_Zna",
    "_Zdl",
    "_Zda",
    # Itanium ABI runtime support: exception machinery, the `__cxxabiv1` class
    # hierarchy vtables, stack protector and TLS helpers.
    "__cxa_",
    "__gxx_",
    "_Unwind_",
    "_ZTVN10__cxxabiv1",
    "_ZTIN10__cxxabiv1",
    "__stack_chk_",
    "__tls_get_addr",
    # SIMD clones the vectorizer emits (`_ZGVbN4v_sin`). Spelled out per ISA and
    # masking letter rather than as a bare `_ZGV`: that would also match
    # `_ZGVZ...`/`_ZGVN...`, the guard variable of a function-local static, and a
    # cross-library reference to one of those is a genuine DT_NEEDED requirement
    # of exactly the kind this comparison exists to catch.
    "_ZGVbN",
    "_ZGVbM",
    "_ZGVcN",
    "_ZGVcM",
    "_ZGVdN",
    "_ZGVdM",
    "_ZGVeN",
    "_ZGVeM",
)

LINUX_IGNORED_UNDEFINED = {
    # `libonedal.so` references this base-class method in `libonedal_core` in the
    # Make build and not in the Bazel one. Both builds define it in
    # `libonedal_core`, and both keep the two derived thunks that call it
    # (`decision_forest::{classification,regression}::internal::ModelImpl::clear`)
    # local. Make's `libonedal.so` additionally carries an unreferenced local copy
    # of those thunks, which drags in the reference to the base; Bazel's link
    # drops the dead copy along with the reference. The exported surface of both
    # trees is identical, so this is a dead-code placement difference rather than
    # a difference in what either release needs from its environment.
    "_ZN4daal10algorithms6dtrees8internal9ModelImpl5clearEv",
}


def run_tool(args):
    result = subprocess.run(
        args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout



def read_linux_dynamic_dependencies(path):
    """The DT_NEEDED list of a shared object, i.e. what its loader must find.

    Exports alone do not describe whether a released library can be loaded. A
    library that resolves `tbb::detail::...` or `std::...` at runtime but records
    no DT_NEEDED entry for the library providing them links fine inside the build
    tree, where the dependency happens to be on the link line of whatever pulls
    it in, and fails in a consumer's process that only loads what the library
    asks for.

    A set, so the comparison is insensitive to DT_NEEDED order. Order does decide
    resolution precedence when two dependencies export the same name, but the two
    build systems order the entries differently on libraries that are otherwise
    in full agreement (libm/libstdc++ trade places), so comparing sequences would
    report a difference on every correct tree.
    """
    # `readelf -d` and `objdump -p` both exit 0 on a file that is not an ELF
    # object, or one with no dynamic section, printing the reason on stdout. An
    # empty dependency set would then compare equal on both sides, so the header
    # is checked here rather than trusting the exit code.
    with open(path, "rb") as handle:
        if handle.read(4) != b"\x7fELF":
            raise RuntimeError(f"not an ELF object: {path}")

    readelf = shutil.which("readelf")
    if readelf:
        output = run_tool([readelf, "-d", str(path)])
        if "no dynamic section" in output:
            raise RuntimeError(f"ELF object has no dynamic section: {path}")
        deps = set()
        for line in output.splitlines():
            if "(NEEDED)" not in line:
                continue
            match = re.search(r"\[([^\]]+)\]", line)
            if match and match.group(1) not in LINUX_IGNORED_NEEDED:
                deps.add(match.group(1))
        return deps

    objdump = shutil.which("objdump")
    if objdump:
        output = run_tool([objdump, "-p", str(path)])
        deps = set()
        for line in output.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "NEEDED":
                if parts[1] not in LINUX_IGNORED_NEEDED:
                    deps.add(parts[1])
        return deps

    raise RuntimeError("neither readelf nor objdump is available")


def is_ignored_linux_undefined(symbol):
    """Whether an undefined dynamic reference is a compiler/libstdc++ artifact."""
    if symbol in LINUX_IGNORED_UNDEFINED:
        return True
    return symbol.startswith(LINUX_IGNORED_UNDEFINED_PREFIXES)


def read_linux_undefined(path):
    """The dynamic symbols a shared object leaves for someone else to define.

    Compared for the same reason as the exports: it is the other half of the
    library's link surface. A reference that exists in one build and not the
    other means the two libraries need different things from their environment,
    which is exactly the difference a DT_NEEDED comparison cannot see on its own
    -- and, read together with the dependencies, it is what tells a maintainer
    whether a missing DT_NEEDED entry is genuinely required.

    Filtered through `is_ignored_linux_undefined`, not through the export ignore
    list: see `LINUX_IGNORED_UNDEFINED_PREFIXES` for why an undefined oneTBB
    reference has to be kept even though an exported one is ignored.

    The `@GLIBC_2.x` / `@GLIBCXX_3.4.x` version suffix is stripped, as the export
    reader already does. That does mean a Bazel library which requires a *newer*
    version of the same symbol -- raising the minimum glibc the release supports
    -- compares equal here; catching that needs a baseline to compare against
    rather than the other release, which is a separate check.
    """
    nm = shutil.which("nm")
    if nm:
        output = run_tool([nm, "-D", "--undefined-only", str(path)])
        symbols = set()
        for line in output.splitlines():
            parts = line.split()
            # `nm -D --undefined-only` prints the binding letter and the name,
            # `U` for a normal undefined reference and `w`/`v` for a weak one.
            if len(parts) == 2 and parts[0] in ("U", "w", "v"):
                symbol = parts[-1].split("@", 1)[0]
                if not is_ignored_linux_undefined(symbol):
                    symbols.add(symbol)
        return symbols

    readelf = shutil.which("readelf")
    if readelf:
        output = run_tool([readelf, "-Ws", str(path)])
        symbols = set()
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[6] == "UND":
                symbol = parts[7].split("@", 1)[0]
                if not is_ignored_linux_undefined(symbol):
                    symbols.add(symbol)
        return symbols

    raise RuntimeError("neither nm nor readelf is available")



def is_shared_library(platform, path):
    if platform == "linux":
        return ".so" in Path(path).name
    suffixes = SHARED_LIBRARY_SUFFIXES[platform]
    return Path(path).suffix.lower() in suffixes



def is_dpc_library(path):
    """Whether `path` names one of the SYCL (DPC++) release libraries.

    Their undefined references are not comparable between the two builds even
    when both are correct: Make links them with `icpx -fsycl ... -lgomp`
    (`dev/make/compiler_definitions/dpcpp.mk`) while Bazel's DPC toolchain uses
    its own flag set, and no CI pairing builds them with the same compiler on
    both sides. The pre-existing export comparison already needs a DPC-specific
    allowance for the same reason (`LINUX_DPC_IGNORED_EXPORT_PREFIXES`); the
    undefined half of the surface is far larger and has no comparable
    allowance, so it is skipped and said out loud rather than filtered by
    guesswork.
    """
    return Path(path).name.startswith(("libonedal_dpc.so", "libonedal_parameters_dpc.so"))



def file_digest(path):
    """Content hash, used to recognise a dereferenced `.so` alias as the same library."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_shared_library_linkage(platform, make_root, bazel_root, files, limit,
                                   cross_toolchain=False, strict_undefined=False):
    """Compare what the released shared libraries *need*, not just what they offer.

    Level 4 used to look at exported symbols only, which made it blind to the one
    property that decides whether a release is usable at all: a library whose
    exports match perfectly and whose DT_NEEDED list is short of
    `libtbb.so.12`/`libstdc++.so.6` still fails to load in a consumer's process.
    Measured on `main` with the same gcc on both sides, every Linux shared object
    in the Bazel release was missing dependencies the Make one records, and the
    comparison reported no differences.

    Returns the error count and, so the staged-dependency check below does not
    have to read every library a second time, the union of the DT_NEEDED entries
    the Make libraries record.

    `cross_toolchain` is for pairings whose two trees are built by different
    compilers -- the nightly compares an icx Make release against a gcc Bazel one.
    Neither half of this check is an equality there, as measured on such a pair:

      * DT_NEEDED. The icx Make libraries record no `libstdc++.so.6` at all,
        because `dev/make/compiler_definitions/icx.mkl.32e.mk` links with the C
        driver `icx` plus `-no-intel-lib` and so never adds it, while the gnu path
        links with `g++`; and they record `libm.so.6` on `libonedal_thread` and
        `libonedal_parameters`, which the `icx` driver adds unconditionally and
        which resolves none of those libraries' undefined symbols. Eight
        differences on four libraries, none of them a Bazel defect.
      * Undefined symbols. 40 symbols differ, 29 of them in the `daal::`
        namespace: icx and gcc inline and place code differently, so which
        references cross a library boundary is a property of the compiler, not of
        the build system. No prefix list can express that.

    So on a cross-toolchain pair the undefined half is skipped and the DT_NEEDED
    half is reported without being counted as an error. Read failures stay errors:
    an unreadable or non-ELF released library is a defect under any pairing.

    `strict_undefined` decides whether a surviving undefined-symbol difference is
    counted. It is off by default, because the undefined half is an equality only
    when both trees resolved the *same* third-party packages, and the Azure
    gcc-vs-gcc pairing does not: Make builds against apt oneMKL 2026.1.0
    (`.ci/env/apt.sh`) while `@mkl` pins conda-forge mkl-static 2025.2.0
    (`MODULE.bazel`). oneDAL links oneMKL statically, so the two releases carry
    different oneMKL objects, and the Bazel libraries reference
    `dlopen`/`dlsym`/`dlclose`/`dlerror`/`printf` where the Make ones do not --
    oneMKL's own static dispatcher, five differences on two libraries, no oneDAL
    defect. With the same oneMKL on both sides the sets match exactly, measured
    locally on a gcc/gcc pair. The DT_NEEDED half is unaffected by the skew and
    stays an equality, and it is the half that catches the defect this check was
    written for: a release that does not record `libtbb.so.12`. Pass
    `--strict-undefined` on a pairing that does control both toolchain and
    dependency packages.
    """
    if platform not in DYNAMIC_LINKAGE_PLATFORMS:
        print(f"Skipped shared library linkage comparison: unsupported on {platform}")
        return 0, set()

    errors = 0
    compared = 0
    dependency_mismatches = []
    undefined_mismatches = []
    unreadable = []
    skipped_undefined = []
    make_dependencies = set()

    # In a tree whose `.so` aliases are real symlinks the aliases never reach
    # `files`, but the Azure jobs download both trees as pipeline artifacts, which
    # dereferences them: `libonedal.so`, `.so.4` and `.so.4.0` all arrive as
    # separate regular files with identical contents. Comparing each of them would
    # read the same library three times and charge one missing DT_NEEDED entry
    # three times over, so a library is compared once per distinct pair of
    # contents. Keying on the contents rather than on the path means an alias that
    # is *not* a copy of its versioned library -- which would be a real defect --
    # is still compared on its own.
    seen = set()

    for path in sorted(files):
        if not is_shared_library(platform, path):
            continue
        make_path = make_root / path
        try:
            identity = (file_digest(make_path), file_digest(bazel_root / path))
        except OSError as err:
            unreadable.append((path, str(err)))
            continue
        if identity in seen:
            continue
        seen.add(identity)
        compared += 1
        compare_undefined = not (cross_toolchain or is_dpc_library(path))
        if not compare_undefined:
            skipped_undefined.append(path)
        # The two reads are kept apart so that a failure in one does not discard
        # the verdict of the other: a DT_NEEDED mismatch already established is
        # the actionable half of the diagnosis and must survive an nm failure.
        try:
            make_needed = read_linux_dynamic_dependencies(make_path)
            # Recorded before the Bazel side is read, so that a library which is
            # unreadable there still contributes its dependencies to the staged
            # runtime check below: losing `libtbb.so.12` from that set would
            # silently drop the unstaged-oneTBB diagnosis.
            make_dependencies |= make_needed
            bazel_needed = read_linux_dynamic_dependencies(bazel_root / path)
        except RuntimeError as err:
            unreadable.append((path, str(err)))
        else:
            if make_needed != bazel_needed:
                dependency_mismatches.append(
                    (path, make_needed - bazel_needed, bazel_needed - make_needed)
                )

        if not compare_undefined:
            continue
        try:
            make_undefined = read_linux_undefined(make_path)
            bazel_undefined = read_linux_undefined(bazel_root / path)
        except RuntimeError as err:
            unreadable.append((path, str(err)))
            continue
        if make_undefined != bazel_undefined:
            undefined_mismatches.append(
                (path, make_undefined - bazel_undefined, bazel_undefined - make_undefined)
            )

    if unreadable:
        errors += len(unreadable)
        print(f"Shared library linkage read failures: {len(unreadable)}")
        for path, err in unreadable[:limit]:
            print(f"  ! {path}: {err}")

    if dependency_mismatches:
        if cross_toolchain:
            print(
                f"Shared library dependency differences: {len(dependency_mismatches)}"
                " (reported only: the two trees are built by different compilers)"
            )
        else:
            errors += len(dependency_mismatches)
            print(f"Shared library dependency mismatches: {len(dependency_mismatches)}")
        for path, only_make, only_bazel in dependency_mismatches[:limit]:
            print(f"  ! {path}: Make-only={len(only_make)}, Bazel-only={len(only_bazel)}")
            for name in sorted(only_make)[:limit]:
                print(f"    - DT_NEEDED {name} recorded by Make, absent from Bazel")
            for name in sorted(only_bazel)[:limit]:
                print(f"    + DT_NEEDED {name} recorded by Bazel, absent from Make")

    if undefined_mismatches:
        if strict_undefined:
            errors += len(undefined_mismatches)
            print(f"Shared library undefined symbol mismatches: {len(undefined_mismatches)}")
        else:
            print(
                f"Shared library undefined symbol differences: {len(undefined_mismatches)}"
                " (reported only: the two trees may resolve different"
                " third-party packages; pass --strict-undefined to gate on this)"
            )
        for path, only_make, only_bazel in undefined_mismatches[:limit]:
            print(f"  ! {path}: Make-only={len(only_make)}, Bazel-only={len(only_bazel)}")
            for symbol in sorted(only_make)[:limit]:
                print(f"    - {symbol}")
            for symbol in sorted(only_bazel)[:limit]:
                print(f"    + {symbol}")

    if skipped_undefined:
        reason = (
            "the two trees are built by different compilers"
            if cross_toolchain
            else "undefined symbols are not comparable across the DPC toolchains"
        )
        print(
            f"Compared dependencies only ({reason}):"
            f" {', '.join(sorted(skipped_undefined))}"
        )

    what = "dependencies" if cross_toolchain else "dependencies and undefined symbols"
    print(f"Compared shared library {what}: {compared}")
    return errors, make_dependencies


def discover_staged_components(root):
    """Sibling component trees of a `<base>/<component>/latest` release root.

    Both builds stage more than oneDAL itself: `makefile:274-279,1101` copies the
    oneTBB runtimes into `<base>/tbb/latest/lib/...`, next to the
    `<base>/daal/latest` this script is pointed at. Because the comparison is
    rooted at the oneDAL component, that whole sibling tree has always been out
    of scope, so a Bazel release that stages no oneTBB at all is indistinguishable
    from one that stages it correctly.

    Returns None when the root does not follow the layout, so the caller can say
    it skipped the check instead of reporting an empty comparison as a pass.
    """
    root = Path(root)
    if root.name != "latest":
        return None
    base = root.parent.parent
    if not base.is_dir():
        return None
    excluded = {root.parent.name}
    components = {}
    try:
        entries = sorted(base.iterdir())
    except OSError as err:
        # An unreadable release root is a problem worth naming, but it is not this
        # check's to diagnose: the tree walk that built `files` runs first and
        # reports it.
        print(f"Skipped staged dependency comparison: cannot read {base}: {err}")
        return None
    for entry in entries:
        if entry.name in excluded or entry.is_symlink() or not entry.is_dir():
            continue
        staged = entry / "latest"
        if staged.is_dir():
            components[entry.name] = staged
    return components


def staged_file_names(roots):
    """Base names of every file staged under the given trees, name -> sample path.

    A name only counts when it resolves to something that exists, so that a
    release staging `libtbb.so.12` as a symlink into a build tree that is no
    longer there is reported as missing rather than accepted. `os.walk` is given
    an `onerror` handler because its default is to yield nothing for a directory
    it cannot read, which would turn a permission problem into a confident and
    wrong "absent from the Bazel release". The handler collects rather than
    raises: an exception here would abort the run and lose the differences
    already found, so the caller charges one error per unreadable directory and
    still prints its summary.
    """
    unreadable = []

    def on_walk_error(err):
        unreadable.append(str(err))

    names = {}
    for root in roots:
        for current, _dirnames, filenames in os.walk(
            root, followlinks=False, onerror=on_walk_error
        ):
            for filename in filenames:
                candidate = Path(current) / filename
                if candidate.exists():
                    names.setdefault(filename, str(candidate))
    return names, unreadable


def compare_staged_dependencies(make_root, bazel_root, dependencies, limit):
    """Check that a dependency Make ships alongside oneDAL is shipped by Bazel too.

    Deliberately narrow, because the two release roots do differ legitimately in
    layout -- Make's is `__release_lnx_gnu/{daal,tbb}/latest`, Bazel's is
    `bazel-bin/release/daal`, with no promise that a component lands in the same
    place. Comparing the sibling trees entry by entry would report that difference
    as dozens of failures on a correct tree. So only one thing is an error: a
    library that

      * a released oneDAL shared object records as a DT_NEEDED dependency, and
      * the Make release stages in a sibling component tree, and
      * does not exist anywhere under the Bazel release root, by that name.

    All three conditions together mean the Bazel release is missing a runtime it
    is required to ship; a component tree that merely sits somewhere else, or a
    staged extra nothing depends on, is reported for information only.
    """
    if not dependencies:
        # The dependency list comes from the linkage comparison, so on a platform
        # where that is skipped this is skipped too -- the filesystem half would
        # work anywhere, but without a dependency list it has no rule to apply.
        # Said out loud, so that a platform where this cannot run yet does not
        # read as a pass. Note the consequence for Windows: a Bazel release that
        # stages no `tbb12.dll` stays undetected until a Windows dependency
        # reader lands.
        print(
            "Skipped staged dependency comparison: no shared library dependency"
            " list (the linkage comparison that produces it did not run)"
        )
        return 0

    if make_root.parent.parent == bazel_root.parent.parent:
        # Both roots would enumerate the same siblings, so every component of one
        # is trivially a component of the other and nothing can ever be reported
        # missing. That is not a pass, it is an unanswerable question: refuse it
        # rather than print a green line. Two real release trees never share a
        # base; a test harness that puts them there has to give them separate
        # parents to exercise this check.
        print(
            "Skipped staged dependency comparison: both release roots share the"
            f" base directory {make_root.parent.parent}, so their staged"
            " components are indistinguishable"
        )
        return 0

    make_components = discover_staged_components(make_root)
    bazel_components = discover_staged_components(bazel_root)
    if make_components is None or bazel_components is None:
        print(
            "Skipped staged dependency comparison: release roots are not"
            " <base>/<component>/latest"
        )
        return 0

    print(
        f"Staged sibling components: Make {sorted(make_components) or '[]'},"
        f" Bazel {sorted(bazel_components) or '[]'}"
    )

    make_staged, make_unreadable = staged_file_names(make_components.values())
    # The compared component tree is included on the Bazel side so a runtime it
    # stages next to the oneDAL libraries instead of in a sibling tree counts as
    # shipped rather than as missing.
    bazel_staged, bazel_unreadable = staged_file_names(
        list(bazel_components.values()) + [bazel_root]
    )

    unreadable = make_unreadable + bazel_unreadable
    errors = len(unreadable)
    if unreadable:
        # Counted rather than raised: an unreadable component tree makes the
        # answer for that tree unknown, and "unknown" has to cost an error, but
        # it must not discard the differences the rest of the run found.
        print(f"Staged component trees that could not be read: {len(unreadable)}")
        for message in unreadable[:limit]:
            print(f"  ! {message}")

    missing = sorted(
        name for name in dependencies
        if name in make_staged and name not in bazel_staged
    )
    errors += len(missing)
    if missing:
        print(f"Staged dependencies missing from Bazel release: {len(missing)}")
        for name in missing[:limit]:
            print(
                f"  ! {name}: required by a released oneDAL library, staged by Make"
                f" at {make_staged[name]}, absent from the Bazel release"
            )
    print(f"Checked staged dependencies: {len(dependencies)}")
    return errors
