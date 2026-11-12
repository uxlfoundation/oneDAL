#===============================================================================
# Copyright 2020 Intel Corporation
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

load("@onedal//dev/bazel:stamp.bzl", "BuildDateInfo", "expand_build_date")
load("@onedal//dev/bazel:utils.bzl", "utils", "paths")
load("@onedal//dev/bazel:cc.bzl", "ModuleInfo")
load("@onedal//dev/bazel/config:config.bzl", "TARGET_ATTRS", "VersionInfo", "target")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

# Transitions get no repo mapping, so they need the canonical label.
_CPU_SETTING = str(Label("@config//:cpu"))

# Both CRT flavours (the debug CRT appends `d`); listed because `onedal_thread` ends in `d`.
_THREAD_STEMS = ["onedal_thread", "onedal_threadd"]

# DLLs that also ship an ABI-versioned import library (`onedal_core_dll.4.lib`), as in Make.
_VERSIONED_IMPLIB_STEMS = [
    "onedal_core",
    "onedal",
    "onedal_dpc",
    "onedal_cored",
    "onedald",
    "onedal_dpcd",
]

def _match_file_name(file, entries):
    # short_path has no bazel-out/ prefix; external files start with "../", which callers can exclude.
    path = file.short_path
    for entry in entries:
        if entry in path:
            return True
    return False

def _collect_headers(dep):
    headers = []
    if ModuleInfo in dep:
        headers += dep[ModuleInfo].compilation_context.headers.to_list()
    elif CcInfo in dep:
        headers += dep[CcInfo].compilation_context.headers.to_list()
    elif DefaultInfo in dep:
        headers += dep[DefaultInfo].files.to_list()
    return utils.unique_files(headers)

def _collect_default_files(deps):
    files = []
    for dep in deps:
        if DefaultInfo in dep:
            files += dep[DefaultInfo].files.to_list()
    return utils.unique_files(files)

def _copy(ctx, src_file, dst_path):
    dst_file = ctx.actions.declare_file(dst_path)
    if target(ctx).is_windows:
        executable = "cmd.exe"
        # `copy` reports "1 file(s) copied." on stdout, which Bazel would echo for every file.
        arguments = ["/d", "/c", 'copy /Y "{}" "{}" >NUL'.format(
            src_file.path.replace("/", "\\"),
            dst_file.path.replace("/", "\\"),
        )]
    else:
        executable = "cp"
        arguments = [src_file.path, dst_file.path]
    ctx.actions.run(
        executable = executable,
        inputs = [src_file],
        outputs = [dst_file],
        arguments = arguments,
        use_default_shell_env = True,
        mnemonic = "CopyFile",
        progress_message = "Copying %s" % dst_file.short_path,
    )
    return dst_file

def _copy_crlf(ctx, src_file, dst_path):
    """Copy `src_file` to `dst_path` with CRLF line endings, as Make's `.release.x` sed.

    The substitutions apply in order, so a CRLF already in the file is not doubled.
    """
    dst_file = ctx.actions.declare_file(dst_path)
    ctx.actions.expand_template(
        template = src_file,
        output = dst_file,
        substitutions = {"\r\n": "\n", "\n": "\r\n"},
    )
    return dst_file

def _try_relativize(path, start):
    if path.startswith(start):
        return paths.relativize(path, start)
    return path

def _copy_version_header(ctx, src_file, dst_path, version_info, is_windows, build_date):
    # The intermediate stays out of the release prefix.
    expanded = ctx.actions.declare_file(ctx.label.name + ".library_version_info.h.in")
    dst_file = ctx.actions.declare_file(dst_path)

    # On Windows Make's `sed` (sed.eol.win) ends only these seven lines with CRLF; match that.
    eol = "\r" if is_windows else ""
    ctx.actions.expand_template(
        template = src_file,
        output = expanded,
        substitutions = {
            "#define __INTEL_DAAL_BUILD_DATE 21990101": "#define __INTEL_DAAL_BUILD_DATE @ONEDAL_BUILD_DATE@{}".format(eol),
            "#define __INTEL_DAAL__        2199": "#define __INTEL_DAAL__ {}{}".format(version_info.major, eol),
            "#define __INTEL_DAAL_MINOR__  9": "#define __INTEL_DAAL_MINOR__ {}{}".format(version_info.minor, eol),
            "#define __INTEL_DAAL_UPDATE__ 9": "#define __INTEL_DAAL_UPDATE__ {}{}".format(version_info.update, eol),
            "#define __INTEL_DAAL_STATUS__ 'A'": "#define __INTEL_DAAL_STATUS__ \"{}\"{}".format(version_info.status, eol),
            "#define __INTEL_DAAL_MAJOR_BINARY__ 999": "#define __INTEL_DAAL_MAJOR_BINARY__ {}{}".format(version_info.binary_major, eol),
            "#define __INTEL_DAAL_MINOR_BINARY__ 999": "#define __INTEL_DAAL_MINOR_BINARY__ {}{}".format(version_info.binary_minor, eol),
        },
    )
    expand_build_date(ctx, expanded, dst_file, build_date)
    return dst_file

def _strip_os_suffix(dst_path, os_suffix):
    """Drop a trailing `_<os>` from the basename, as Make's `$(subst _$(_OS),,$d)`."""
    base = paths.basename(dst_path)
    stem, _, extension = base.rpartition(".")
    if not stem or not stem.endswith(os_suffix):
        return dst_path
    stripped = "{}.{}".format(stem[:-len(os_suffix)], extension)
    return paths.join(paths.dirname(dst_path), stripped)

def _copy_include(ctx, prefix, version_info):
    include_prefix = paths.join(prefix, "include")
    build_date = ctx.attr._build_date[BuildDateInfo]
    is_windows = target(ctx).is_windows
    os_suffix = "_win" if is_windows else "_lnx"

    # One header per destination; an OS-specific header wins over the generic one, as in Make.
    staged_order = []
    staged = {}
    os_specific = {}
    for include, prefix, skip_prefix in zip(ctx.attr.include, ctx.attr.include_prefix,
                                            ctx.attr.include_skip_prefix):
        headers = _collect_headers(include)
        for header in headers:
            if header.basename == "_dal_cpu_dispatcher_gen.hpp":
                continue
            if skip_prefix:
                # file.path has a bazel-out/<cfg>/ prefix that skip_prefix would not match.
                dst_path = _try_relativize(header.short_path, skip_prefix)
            elif prefix:
                dst_path = paths.join(prefix, header.basename)
            dst_path = paths.join(include_prefix, dst_path)
            stripped = _strip_os_suffix(dst_path, os_suffix)
            is_os_specific = stripped != dst_path
            dst_path = stripped
            if dst_path not in staged:
                staged_order.append(dst_path)
            elif os_specific.get(dst_path) and not is_os_specific:
                continue
            staged[dst_path] = header
            if is_os_specific:
                os_specific[dst_path] = True

    dst_files = []
    for dst_path in staged_order:
        header = staged[dst_path]
        if header.short_path == "cpp/daal/include/services/library_version_info.h":
            dst_file = _copy_version_header(ctx, header, dst_path, version_info,
                                            is_windows, build_date)
        else:
            dst_file = _copy(ctx, header, dst_path)
        dst_files.append(dst_file)
    return dst_files

def _symlink(ctx, link_name, target_name, prefix):
    """Create a relative symlink in the release directory.

    Args:
        ctx: rule context.
        link_name: basename of the symlink, e.g. `libonedal_core.so`.
        target_name: basename of the target in the same directory.
        prefix: directory of the symlink in the rule's output tree.
    """
    link_file = ctx.actions.declare_symlink(paths.join(prefix, link_name))
    ctx.actions.symlink(
        output = link_file,
        target_path = target_name,
    )
    return link_file

def _windows_dll_stem(name, version_info):
    """`onedal_core` for `onedal_core.4`: released import library names carry no ABI version."""
    suffix = "." + version_info.binary_major
    return name[:-len(suffix)] if name.endswith(suffix) else name

def _copy_lib(ctx, prefix, version_info):
    """Copy the libraries into the release tree, with SONAME symlinks for each Linux `.so`.

    A DLL's `<name>.<major>.if.lib` is staged under Make's `-IMPLIB:` name, `<name>_dll.lib`.
    """
    is_windows = target(ctx).is_windows
    arch_dir = target(ctx).arch_dir
    lib_prefix = paths.join(prefix, "lib", arch_dir)
    redist_prefix = paths.join(prefix, "redist", arch_dir)
    libs = _collect_default_files(ctx.attr.lib)
    dst_files = []

    if is_windows:
        implib_stems = [
            lib.basename[:-len(".if.lib")]
            for lib in libs
            if lib.basename.endswith(".if.lib")
        ]
        for lib in libs:
            if lib.extension == "dll" and lib.basename[:-len(".dll")] not in implib_stems:
                fail("{} has no import library: the Windows toolchain must emit {}.if.lib".format(
                    lib.basename, lib.basename[:-len(".dll")]))

    for lib in libs:
        # macOS .dylib files are not versioned yet.
        is_shared_lib = lib.extension == "so"

        if is_shared_lib:
            binary_major = version_info.binary_major
            binary_minor = version_info.binary_minor

            # 1. Real versioned file: libonedal_core.so.2.9
            versioned_name = "{}.{}.{}".format(lib.basename, binary_major, binary_minor)
            versioned_file = _copy(ctx, lib, paths.join(lib_prefix, versioned_name))
            dst_files.append(versioned_file)

            # 2. Major symlink: libonedal_core.so.2 -> libonedal_core.so.2.9
            major_link_name = "{}.{}".format(lib.basename, binary_major)
            dst_files.append(_symlink(ctx, major_link_name, versioned_name, lib_prefix))

            # 3. Unversioned symlink: libonedal_core.so -> libonedal_core.so.2
            dst_files.append(_symlink(ctx, lib.basename, major_link_name, lib_prefix))
        elif is_windows and lib.extension == "dll":
            # The Make layout keeps the DLLs under redist/<arch>, not lib/<arch>.
            dst_files.append(_copy(ctx, lib, paths.join(redist_prefix, lib.basename)))
        elif is_windows and lib.basename.endswith(".if.lib"):
            stem = _windows_dll_stem(lib.basename[:-len(".if.lib")], version_info)
            # The thread import library is not shipped in the Make package.
            if stem in _THREAD_STEMS:
                continue
            dst_files.append(_copy(ctx, lib, paths.join(lib_prefix, "{}_dll.lib".format(stem))))
            if stem in _VERSIONED_IMPLIB_STEMS:
                dst_files.append(_copy(ctx, lib, paths.join(
                    lib_prefix,
                    "{}_dll.{}.lib".format(stem, version_info.binary_major),
                )))
        else:
            dst_files.append(_copy(ctx, lib, paths.join(lib_prefix, lib.basename)))

    return dst_files

# CRLF in the Make Windows package: Make's `.release.x` sed rewrites config.txt and data/, and
# cmake's configure_file writes host newlines. vars.bat is CRLF in the repository already.
_CRLF_EXTRA_FILES = [
    "config/config.txt",
    "lib/cmake/oneDAL/oneDALConfig.cmake",
    "lib/cmake/oneDAL/oneDALConfigVersion.cmake",
]

# Make's `expat` list for data/. Examples and samples keep LF in both packages.
_CRLF_DATA_SUFFIXES = [".cmake", ".cpp", ".csv", ".h", ".hpp", ".txt"]

def _is_crlf_staged(dst_subpath):
    """True when the Make Windows package ships `dst_subpath` with CRLF."""
    if dst_subpath in _CRLF_EXTRA_FILES:
        return True
    if not dst_subpath.startswith("data/"):
        return False
    for suffix in _CRLF_DATA_SUFFIXES:
        if dst_subpath.endswith(suffix):
            return True
    return False

def _copy_extra_files(ctx, prefix):
    """Copy each `extra_files` entry to its `extra_files_dst` path under the release root."""
    if len(ctx.attr.extra_files) != len(ctx.attr.extra_files_dst):
        fail("extra_files and extra_files_dst must have the same length: got {} vs {}".format(
            len(ctx.attr.extra_files), len(ctx.attr.extra_files_dst)))

    is_windows = target(ctx).is_windows

    dst_files = []
    for i, dep in enumerate(ctx.attr.extra_files):
        dst_subpath = ctx.attr.extra_files_win_dst[i] if is_windows and ctx.attr.extra_files_win_dst[i] != None else ctx.attr.extra_files_dst[i]
        if not dst_subpath:
            continue
        srcs = dep[DefaultInfo].files.to_list()
        if len(srcs) != 1:
            fail("extra_files entry '{}' must produce exactly one file, got {}".format(
                dep.label, len(srcs)))
        src = srcs[0]
        dst_path = paths.join(prefix, dst_subpath)
        if is_windows and _is_crlf_staged(dst_subpath):
            dst_files.append(_copy_crlf(ctx, src, dst_path))
        else:
            dst_files.append(_copy(ctx, src, dst_path))
    return dst_files

def _copy_dep_runtime(ctx, release_root):
    """Stage third-party runtimes (TBB) next to `daal/latest`; paths are relative to its parent."""
    if len(ctx.attr.dep_runtime) != len(ctx.attr.dep_runtime_dst):
        fail("dep_runtime and dep_runtime_dst must have the same length: got {} vs {}".format(
            len(ctx.attr.dep_runtime), len(ctx.attr.dep_runtime_dst)))

    is_windows = target(ctx).is_windows

    dst_files = []
    for i, dep in enumerate(ctx.attr.dep_runtime):
        dst_dir = ctx.attr.dep_runtime_win_dst[i] if is_windows else ctx.attr.dep_runtime_dst[i]
        if not dst_dir:
            continue
        if dst_dir == "daal" or dst_dir.startswith("daal/"):
            fail(("dep_runtime destination '{}' of {} would write into the " +
                  "daal/latest tree, which the release rule stages itself").format(
                dst_dir, dep.label))
        srcs = dep[DefaultInfo].files.to_list()
        if not srcs:
            # Make fails here too; the repository rules glob with `allow_empty = True`.
            fail("dep_runtime target {} produced no files to stage into '{}'".format(
                dep.label, dst_dir))
        for src in srcs:
            dst_path = paths.join(release_root, dst_dir, src.basename)
            dst_files.append(_copy(ctx, src, dst_path))
    return dst_files

def _copy_data(ctx, prefix):
    """Copy data files (datasets, examples, config) preserving directory structure."""
    is_windows = target(ctx).is_windows

    dst_files = []
    for dep in ctx.attr.data:
        srcs = dep[DefaultInfo].files.to_list()
        for src in srcs:
            if src.short_path.startswith("../"):
                continue
            dst_path = paths.join(prefix, src.short_path)
            if is_windows and _is_crlf_staged(src.short_path):
                dst_files.append(_copy_crlf(ctx, src, dst_path))
            else:
                dst_files.append(_copy(ctx, src, dst_path))
    return dst_files

def _copy_to_release_impl(ctx):
    prefix = ctx.attr.name + "/daal/latest"
    version_info = ctx.attr._version_info[VersionInfo]
    files = []
    files += _copy_include(ctx, prefix, version_info)
    files += _copy_lib(ctx, prefix, version_info)
    files += _copy_extra_files(ctx, prefix)
    files += _copy_data(ctx, prefix)
    files += _copy_dep_runtime(ctx, ctx.attr.name)
    return [DefaultInfo(files=depset(files))]

def _release_cpu_all_transition_impl(settings, attr):
    """Build every ISA unless --cpu is set; an explicit --cpu=auto also becomes 'all'."""
    current = settings[_CPU_SETTING]
    if current == "auto":
        return {_CPU_SETTING: "all"}
    return {_CPU_SETTING: current}

_release_cpu_all_transition = transition(
    implementation = _release_cpu_all_transition_impl,
    inputs = [_CPU_SETTING],
    outputs = [_CPU_SETTING],
)

_release = rule(
    implementation = _copy_to_release_impl,
    attrs = {
        "include": attr.label_list(allow_files=True, cfg=_release_cpu_all_transition),
        "include_prefix": attr.string_list(),
        "include_skip_prefix": attr.string_list(),
        "lib": attr.label_list(allow_files=True, cfg=_release_cpu_all_transition),
        "data": attr.label_list(allow_files=True, cfg=_release_cpu_all_transition),
        "extra_files": attr.label_list(
            allow_files = False,
            doc = "Additional generated files to include in release. Must be rule targets (not bare file labels). Must be paired 1:1 with extra_files_dst.",
        ),
        "extra_files_dst": attr.string_list(
            doc = "Destination paths for extra_files, relative to the release root.",
        ),
        "extra_files_win_dst": attr.string_list(
            doc = "Windows-specific destination paths for extra_files; empty skips the file on Windows.",
        ),
        "dep_runtime": attr.label_list(
            allow_files = False,
            doc = "Third-party runtime libraries staged next to `daal/latest`. " +
                  "Must be rule targets (not bare file labels), like extra_files. " +
                  "Must be paired 1:1 with dep_runtime_dst.",
        ),
        "dep_runtime_dst": attr.string_list(
            doc = "Destination directory of each dep_runtime entry, relative to the release root.",
        ),
        "dep_runtime_win_dst": attr.string_list(
            doc = "Windows-specific destination directory of each dep_runtime " +
                  "entry; empty skips the entry on Windows.",
        ),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
        "_build_date": attr.label(
            default = "@onedal//dev/bazel:build_date",
            providers = [BuildDateInfo],
        ),
    } | TARGET_ATTRS,
)

def _headers_filter_impl(ctx):
    all_headers = []
    for dep in ctx.attr.deps:
        all_headers += _collect_headers(dep)
    all_headers = utils.unique_files(all_headers)
    filtered_headers = []
    for header in all_headers:
        if (_match_file_name(header, ctx.attr.include) and
            not _match_file_name(header, ctx.attr.exclude)):
            filtered_headers.append(header)
    return [
        DefaultInfo(files = depset(filtered_headers))
    ]


headers_filter = rule(
    implementation = _headers_filter_impl,
    attrs = {
        "deps": attr.label_list(allow_files=True),
        "include": attr.string_list(),
        "exclude": attr.string_list(),
    },
)

_MSVC_RUNTIME_SETTING = str(Label("@config//:msvc_runtime"))

# The only CRT-dependent directories; the rest comes from the release-CRT build. Narrower than
# `lib/`, whose pkgconfig and cmake files have the same names in both flavours.
def _runtime_specific_dirs(arch_dir):
    return ["lib/{}/".format(arch_dir), "redist/{}/".format(arch_dir)]

# Pin the MSVC runtime, ignoring any `--msvc_runtime` on the command line.
def _force_msvc_runtime_release_impl(settings, attr):
    return {_MSVC_RUNTIME_SETTING: "release"}

def _force_msvc_runtime_debug_impl(settings, attr):
    return {_MSVC_RUNTIME_SETTING: "debug"}

_force_msvc_runtime_release_transition = transition(
    implementation = _force_msvc_runtime_release_impl,
    inputs = [],
    outputs = [_MSVC_RUNTIME_SETTING],
)

_force_msvc_runtime_debug_transition = transition(
    implementation = _force_msvc_runtime_debug_impl,
    inputs = [],
    outputs = [_MSVC_RUNTIME_SETTING],
)

def _relativize_release_file(file, dep):
    """Split `file` into (`<component>/latest`, rest) under `dep`'s release root, or None."""
    path = file.short_path
    marker = dep.label.name + "/"
    start = 0
    for _ in range(path.count(marker)):
        index = path.find(marker, start)
        if index == -1:
            break
        start = index + len(marker)
        parts = path[start:].split("/")
        # Otherwise a package directory that shares the target's name.
        if len(parts) >= 3 and parts[1] == "latest":
            return (paths.join(parts[0], "latest"), "/".join(parts[2:]))
    return None

def _copy_from_release_tree(ctx, dep, prefix, only_dirs = None, staged = None):
    """Copy `dep`'s release tree under `prefix` (the `daal/latest` tree).

    `only_dirs` filters `daal/latest` only: sibling components such as TBB come from both
    CRT flavours, whose runtimes have distinct names. `staged` holds the destinations
    already declared, since Bazel rejects a second `declare_file` for one path.
    """
    dst_files = []
    release_root = paths.dirname(paths.dirname(prefix))
    for src in dep[DefaultInfo].files.to_list():
        relative = _relativize_release_file(src, dep)
        if relative == None:
            fail("Unexpected file '{}' in release tree of {}".format(
                src.short_path, dep.label))
        component, rel = relative
        if component == "daal/latest":
            if only_dirs != None:
                if not [d for d in only_dirs if rel.startswith(d)]:
                    continue
            dst = paths.join(prefix, rel)
        else:
            dst = paths.join(release_root, component, rel)
            if staged != None:
                previous = staged.get(dst)
                if previous != None:
                    if previous != src.short_path:
                        fail(("Release component file '{}' of {} is staged from " +
                              "both '{}' and '{}': the two MSVC runtime flavours " +
                              "disagree on what the merged tree should hold").format(
                            dst, dep.label, previous, src.short_path))
                    continue
                staged[dst] = src.short_path
        dst_files.append(_copy(ctx, src, dst))
    return dst_files

def _get_single_dep(deps, attr_name):
    """Return the one configured target of a transitioned label_list attribute."""
    if len(deps) != 1:
        fail("Attribute '{}' must hold exactly one target, got {}".format(
            attr_name, len(deps)))
    return deps[0]

def _release_all_impl(ctx):
    release_md = _get_single_dep(ctx.attr.release_md, "release_md")
    if not target(ctx).is_windows:
        # Forwarded rather than copied: copying would turn the `.so` symlinks into real files.
        return [DefaultInfo(files = release_md[DefaultInfo].files)]
    prefix = ctx.attr.name + "/daal/latest"
    staged_components = {}
    files = _copy_from_release_tree(ctx, release_md, prefix, staged = staged_components)
    # Only the `d`-suffixed libraries differ; pkg-config describes the release CRT, and
    # oneDALConfig.cmake appends DAL_DEBUG_SUFFIX itself.
    files += _copy_from_release_tree(
        ctx, _get_single_dep(ctx.attr.release_mdd, "release_mdd"), prefix,
        only_dirs = _runtime_specific_dirs(target(ctx).arch_dir),
        staged = staged_components,
    )
    return [DefaultInfo(files = depset(files))]

_release_all = rule(
    implementation = _release_all_impl,
    attrs = {
        "release_md": attr.label_list(
            mandatory = True,
            cfg = _force_msvc_runtime_release_transition,
            doc = "Release tree built against the release MSVC runtime.",
        ),
        "release_mdd": attr.label_list(
            mandatory = True,
            cfg = _force_msvc_runtime_debug_transition,
            doc = "Release tree built against the debug MSVC runtime. " +
                  "Only its lib/<arch> and redist/<arch> contents are " +
                  "used; ignored entirely on non-Windows platforms.",
        ),
    } | TARGET_ATTRS,
)

def release_all(name, release_target):
    """Build `release_target` with `-MD` and `-MDd` and merge both into one tree (Windows only)."""
    _release_all(
        name = name,
        release_md = [release_target],
        release_mdd = [release_target],
    )

def release_include(hdrs, skip_prefix="", add_prefix=""):
    return (hdrs, add_prefix, skip_prefix)

def release(name, include, lib, extra_files = [], data = [], dep_runtime = []):
    """Assemble the oneDAL release directory tree.

    Args:
        name: target name and output directory prefix.
        include: release_include() entries.
        lib: library targets.
        extra_files: release_extra_file() entries.
        data: filegroups copied with their directory structure.
        dep_runtime: release_dep_runtime() entries, staged next to `daal/latest`.
    """
    rule_include = []
    rule_include_prefix = []
    rule_include_skip_prefix = []
    for hdrs, prefix, skip_prefix in include:
        for dep in hdrs:
            rule_include.append(dep)
            rule_include_prefix.append(prefix)
            rule_include_skip_prefix.append(skip_prefix)

    rule_extra_files = []
    rule_extra_files_dst = []
    rule_extra_files_win_dst = []
    for extra_file in extra_files:
        if len(extra_file) == 2:
            label, dst = extra_file
            win_dst = dst
        else:
            label, dst, win_dst = extra_file
        rule_extra_files.append(label)
        rule_extra_files_dst.append(dst)
        rule_extra_files_win_dst.append(win_dst)

    _release(
        name = name,
        include = rule_include,
        include_prefix = rule_include_prefix,
        include_skip_prefix = rule_include_skip_prefix,
        lib = lib,
        data = data,
        extra_files = rule_extra_files,
        extra_files_dst = rule_extra_files_dst,
        extra_files_win_dst = rule_extra_files_win_dst,
        dep_runtime = [entry[0] for entry in dep_runtime],
        dep_runtime_dst = [entry[1] for entry in dep_runtime],
        dep_runtime_win_dst = [entry[2] if len(entry) == 3 else entry[1] for entry in dep_runtime],
    )

def release_extra_file(label, dst_path, windows_dst_path = None):
    """Declare an extra file for release().

    Args:
        label: target producing the file.
        dst_path: destination relative to the release root, e.g. `env/vars.sh`.
        windows_dst_path: Windows destination; defaults to `dst_path`, "" skips the file.
    """
    return (label, dst_path, dst_path if windows_dst_path == None else windows_dst_path)

def release_dep_runtime(label, dst_dir, windows_dst_dir = None):
    """Declare a staged runtime dependency for release().

    Args:
        label: target producing the files.
        dst_dir: destination relative to the directory holding `daal/latest`, e.g. `tbb/latest/lib`.
        windows_dst_dir: Windows destination; defaults to `dst_dir`, "" skips the entry.
    """
    return (label, dst_dir, dst_dir if windows_dst_dir == None else windows_dst_dir)
