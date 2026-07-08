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

import argparse
import filecmp
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path


TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".cmake",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    ".in",
    ".lst",
    ".pc",
    ".sh",
    ".txt",
    ".xml",
}

TEXT_NAMES = {
    "CMakeLists.txt",
    "license.txt",
    "makefile",
}

BINARY_SUFFIXES = {
    ".a",
    ".dll",
    ".exp",
    ".lib",
    ".pdb",
    ".so",
}

SHARED_LIBRARY_SUFFIXES = {
    "linux": (".so",),
    "windows": (".dll",),
}

IGNORED_FILES = {
    "linux": {
        # Make DPC release does not publish these static DPC archives; keep the
        # comparison focused on the shared libraries and public package surface.
    },
}

NORMALIZED_TEXT_LINES = {
    "include/services/library_version_info.h": (
        "__INTEL_DAAL_BUILD_DATE",
    ),
}

LINUX_IGNORED_EXPORTS = {
    # ELF linker-defined section boundary symbols. These are not part of the
    # oneDAL ABI, but GNU nm reports them as dynamic definitions for some
    # shared objects.
    "__bss_start",
    "_edata",
    "_end",
    # Intel compiler/runtime math symbols may be pulled into Bazel-built shared
    # objects while Make hides them with linker options. They are not oneDAL API.
}

LINUX_IGNORED_EXPORT_PREFIXES = (
    # Implementation details instantiated through oneTBB/libstdc++ may vary
    # between Make and Bazel builds while keeping the public oneDAL ABI intact.
    "_ZN3tbb6detail",
    "_ZNK3tbb6detail",
    "_ZSt27__unguarded_partition_pivot",
    "_Z28_daal_parallel_sort_template",
    "_ZN4daal10algorithms8internal5qSort",
    "MKL_",
    "mkl_",
)

LINUX_DPC_IGNORED_EXPORT_PREFIXES = (
    # Bazel may expose symbols from static dependencies linked into the DPC
    # library that Make keeps hidden. These symbols do not describe the public
    # DPC package surface being validated by this comparison.
    "_Z22__daal_serv_cpu_detect",
    "_Z23daal_check_is_intel_cpu",
    "_Z23daal_enabled_cpu_detect",
    "_Z28daal_serv_cpu_feature_detect",
    "_ZN21mkl_lapack_tbb_compat",
    "_ZN4daal",
    "_ZNK4daal",
    "_ZTIN4daal",
    "_ZTSN4daal",
    "_ZTVN4daal",
)


def is_ignored_linux_export(symbol, library_path=None):
    if symbol in LINUX_IGNORED_EXPORTS:
        return True
    if symbol.startswith(LINUX_IGNORED_EXPORT_PREFIXES):
        return True
    if library_path and Path(library_path).name.startswith("libonedal_dpc.so"):
        if symbol.startswith(LINUX_DPC_IGNORED_EXPORT_PREFIXES):
            return True
        if re.match(r"^_ZNK\d+mkl_sparse_", symbol):
            return True
    # BLAS/LAPACK entry points from MKL are uppercase Fortran-style names.
    return bool(re.match(r"^[A-Z][A-Z0-9_]+(_64)?$", symbol))


def rel(path, root):
    return path.relative_to(root).as_posix()


def classify(root):
    dirs = set()
    files = set()
    links = {}

    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            path = current_path / dirname
            rel_path = rel(path, root)
            if path.is_symlink():
                links[rel_path] = os.readlink(path)
            else:
                dirs.add(rel_path)
        for filename in filenames:
            path = current_path / filename
            rel_path = rel(path, root)
            if path.is_symlink():
                links[rel_path] = os.readlink(path)
            else:
                files.add(rel_path)

    return dirs, files, links


def is_text_path(path):
    p = Path(path)
    if p.name in TEXT_NAMES:
        return True
    if p.suffix.lower() in TEXT_SUFFIXES:
        return True
    if p.suffix.lower() in BINARY_SUFFIXES:
        return False
    return False


def read_normalized_text(root, path):
    content = (root / path).read_text(encoding="utf-8", errors="surrogateescape")
    prefixes = NORMALIZED_TEXT_LINES.get(path, ())
    if not prefixes:
        return content
    normalized = []
    for line in content.splitlines(keepends=True):
        stripped = line.lstrip()
        if any(stripped.startswith(f"#define {prefix} ") for prefix in prefixes):
            newline = "\n" if line.endswith("\n") else ""
            normalized.append(f"#define {stripped.split()[1]} <normalized>{newline}")
        else:
            normalized.append(line)
    return "".join(normalized)


def compare_sets(name, make_values, bazel_values, limit):
    errors = 0
    only_make = sorted(make_values - bazel_values)
    only_bazel = sorted(bazel_values - make_values)

    if only_make:
        errors += len(only_make)
        print(f"Only Make {name}: {len(only_make)}")
        for item in only_make[:limit]:
            print(f"  - {item}")
    if only_bazel:
        errors += len(only_bazel)
        print(f"Only Bazel {name}: {len(only_bazel)}")
        for item in only_bazel[:limit]:
            print(f"  + {item}")
    return errors


def compare_link_targets(make_links, bazel_links, limit):
    errors = 0
    common_links = set(make_links).intersection(bazel_links)
    bad_links = [
        path for path in sorted(common_links)
        if make_links[path] != bazel_links[path]
    ]
    if bad_links:
        errors += len(bad_links)
        print(f"Symlink target mismatches: {len(bad_links)}")
        for path in bad_links[:limit]:
            print(f"  ! {path}: Make -> {make_links[path]}, Bazel -> {bazel_links[path]}")
    print(f"Compared symlink targets: {len(common_links)}")
    return errors


def compare_text_files(make_root, bazel_root, files, limit):
    errors = 0
    mismatches = []
    compared = 0

    for path in sorted(files):
        if not is_text_path(path):
            continue
        compared += 1
        make_path = make_root / path
        bazel_path = bazel_root / path
        if path in NORMALIZED_TEXT_LINES:
            equal = read_normalized_text(make_root, path) == read_normalized_text(bazel_root, path)
        else:
            equal = filecmp.cmp(make_path, bazel_path, shallow=False)
        if not equal:
            mismatches.append(path)

    if mismatches:
        errors += len(mismatches)
        print(f"Text content mismatches: {len(mismatches)}")
        for item in mismatches[:limit]:
            print(f"  ! {item}")
    print(f"Compared text files: {compared}")
    return errors


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


def read_linux_exports(path):
    nm = shutil.which("nm")
    if nm:
        output = run_tool([nm, "-D", "--defined-only", str(path)])
        symbols = set()
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                symbol = parts[-1].split("@", 1)[0]
                if not is_ignored_linux_export(symbol, path):
                    symbols.add(symbol)
        return symbols

    readelf = shutil.which("readelf")
    if readelf:
        output = run_tool([readelf, "-Ws", str(path)])
        symbols = set()
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[3] == "FUNC" and parts[6] != "UND":
                symbol = parts[7].split("@", 1)[0]
                if not is_ignored_linux_export(symbol, path):
                    symbols.add(symbol)
        return symbols

    raise RuntimeError("neither nm nor readelf is available")


def read_windows_exports(path):
    dumpbin = shutil.which("dumpbin")
    if dumpbin:
        output = run_tool([dumpbin, "/NOLOGO", "/EXPORTS", str(path)])
        symbols = set()
        in_exports = False
        for line in output.splitlines():
            if "ordinal" in line and "hint" in line and "RVA" in line:
                in_exports = True
                continue
            if not in_exports:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("Summary"):
                continue
            parts = stripped.split()
            if len(parts) >= 4 and parts[0].isdigit():
                symbols.add(parts[3].split("=", 1)[0])
        return symbols

    llvm_nm = shutil.which("llvm-nm")
    if llvm_nm:
        output = run_tool([llvm_nm, "--defined-only", "--extern-only", str(path)])
        symbols = set()
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 3 and re.match(r"^[A-Za-z]$", parts[-2]):
                symbols.add(parts[-1])
        return symbols

    raise RuntimeError("neither dumpbin nor llvm-nm is available")


def read_exports(platform, path):
    if platform == "linux":
        return read_linux_exports(path)
    if platform == "windows":
        return read_windows_exports(path)
    raise ValueError(f"unsupported platform: {platform}")


def is_shared_library(platform, path):
    if platform == "linux":
        return ".so" in Path(path).name
    suffixes = SHARED_LIBRARY_SUFFIXES[platform]
    return Path(path).suffix.lower() in suffixes


def compare_shared_library_exports(platform, make_root, bazel_root, files, limit):
    errors = 0
    compared = 0
    mismatches = []
    unreadable = []

    for path in sorted(files):
        if not is_shared_library(platform, path):
            continue
        compared += 1
        make_path = make_root / path
        bazel_path = bazel_root / path
        try:
            make_exports = read_exports(platform, make_path)
            bazel_exports = read_exports(platform, bazel_path)
        except RuntimeError as err:
            unreadable.append((path, str(err)))
            continue
        only_make = make_exports - bazel_exports
        only_bazel = bazel_exports - make_exports
        if only_make or only_bazel:
            mismatches.append((path, only_make, only_bazel))

    if unreadable:
        errors += len(unreadable)
        print(f"Shared library export read failures: {len(unreadable)}")
        for path, err in unreadable[:limit]:
            print(f"  ! {path}: {err}")

    if mismatches:
        errors += len(mismatches)
        print(f"Shared library export mismatches: {len(mismatches)}")
        for path, only_make, only_bazel in mismatches[:limit]:
            print(f"  ! {path}: Make-only={len(only_make)}, Bazel-only={len(only_bazel)}")
            for symbol in sorted(only_make)[:limit]:
                print(f"    - {symbol}")
            for symbol in sorted(only_bazel)[:limit]:
                print(f"    + {symbol}")

    print(f"Compared shared library exports: {compared}")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Compare Make and Bazel oneDAL release trees.",
    )
    parser.add_argument("--make", required=True, type=Path)
    parser.add_argument("--bazel", required=True, type=Path)
    parser.add_argument("--platform", choices=("linux", "windows"), required=True)
    parser.add_argument("--check-level", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--summary-limit", type=int, default=50)
    args = parser.parse_args()
    if args.structure_only:
        args.check_level = 1

    make_root = args.make.resolve()
    bazel_root = args.bazel.resolve()

    if not make_root.is_dir():
        print(f"Make release directory not found: {make_root}", file=sys.stderr)
        return 2
    if not bazel_root.is_dir():
        print(f"Bazel release directory not found: {bazel_root}", file=sys.stderr)
        return 2

    print(f"Platform: {args.platform}")
    print(f"Make release:  {make_root}")
    print(f"Bazel release: {bazel_root}")

    make_dirs, make_files, make_links = classify(make_root)
    bazel_dirs, bazel_files, bazel_links = classify(bazel_root)

    ignored_files = IGNORED_FILES.get(args.platform, set())
    make_files -= ignored_files
    bazel_files -= ignored_files

    print(f"Make:  {len(make_dirs)} dirs, {len(make_files)} files, {len(make_links)} links")
    print(f"Bazel: {len(bazel_dirs)} dirs, {len(bazel_files)} files, {len(bazel_links)} links")

    print(f"Check level: {args.check_level}")

    errors = 0
    print("")
    print("=== level 1: release tree entries ===")
    errors += compare_sets("directories", make_dirs, bazel_dirs, args.summary_limit)
    errors += compare_sets("files", make_files, bazel_files, args.summary_limit)
    errors += compare_sets("symlinks", set(make_links), set(bazel_links), args.summary_limit)

    if args.check_level >= 2:
        print("")
        print("=== level 2: symlink targets ===")
        errors += compare_link_targets(make_links, bazel_links, args.summary_limit)

    common_files = make_files.intersection(bazel_files)
    if args.check_level >= 3:
        print("")
        print("=== level 3: text file contents ===")
        errors += compare_text_files(make_root, bazel_root, common_files, args.summary_limit)

    if args.check_level >= 4:
        print("")
        print("=== level 4: shared library exports ===")
        errors += compare_shared_library_exports(
            args.platform,
            make_root,
            bazel_root,
            common_files,
            args.summary_limit,
        )

    if errors:
        print(f"Release comparison failed: {errors} difference(s)")
        return 1

    print("Release comparison passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
