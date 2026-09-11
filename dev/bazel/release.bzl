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

load("@onedal//dev/bazel:utils.bzl", "utils", "paths")
load("@onedal//dev/bazel:cc.bzl", "ModuleInfo")
load("@onedal//dev/bazel/config:config.bzl", "VersionInfo")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

# Resolve @config//:cpu to its Bzlmod canonical label at load time.
# Using a plain string "@config//:cpu" inside a transition fails under Bzlmod
# because repo mapping is not applied in that context.
_CPU_SETTING = str(Label("@config//:cpu"))

# Windows library stems, in both MSVC-runtime flavours: a debug-CRT build
# appends `d` to every library name (see `_msvc_runtime_suffix` in
# dev/bazel/cc.bzl and `$d` in makefile:124). Listed explicitly rather than
# stripping a trailing `d`, because `onedal_thread` already ends in one.
_THREAD_STEMS = ["onedal_thread", "onedal_threadd"]

# DLL stems that additionally ship an ABI-versioned import library
# (`onedal_core_dll.4.lib`), matching `.release.a_win` in makefile:962-970.
_VERSIONED_IMPLIB_STEMS = [
    "onedal_core",
    "onedal",
    "onedal_dpc",
    "onedal_cored",
    "onedald",
    "onedal_dpcd",
]

def _match_file_name(file, entries):
    # Use short_path (workspace-relative, e.g. "cpp/oneapi/dal/foo.h" or
    # "../mkl_repo~mkl/include/mkl.h") instead of file.path (which contains
    # bazel-out/ prefixes and Bzlmod repo hashes like "+mkl_repo+mkl/").
    # This makes external-repo detection stable across workspace and Bzlmod:
    #   - workspace style:  short_path starts with "../<repo>/"
    #   - Bzlmod style:     short_path starts with "../<repo>~<ver>/"
    # Callers can use "../" as an exclude pattern to reject all external repos.
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

def _make_implib_for_dll(ctx, dll_file, lib_dst_path):
    """Derive a Windows DLL's import library at `lib_dst_path` from `dll_file`.

    Runs `dev/bazel/toolchains/tools/dll_to_implib.bat` which dumps the
    DLL's exports table and feeds it to `lib /def:` to produce a fresh
    `.lib`. Mirrors the makefile's `-IMPLIB:<name>_dll.lib` linker flag,
    which we cannot pass directly through Bazel's cc_common.link()
    because there is no API to register the side-effect file as an
    action output. Generating it from the DLL post-link is equivalent.
    """
    lib_file = ctx.actions.declare_file(lib_dst_path)
    # `lib /def:` requires an .exp output, but it is an intermediate rather
    # than a release artifact. Keep it outside the staged release tree.
    exp_file = ctx.actions.declare_file(paths.join(
        "_release_intermediates",
        lib_file.basename[:-len(".lib")] + ".exp",
    ))
    script = ctx.file._dll_to_implib
    ctx.actions.run(
        executable = "cmd.exe",
        inputs = [dll_file, script],
        outputs = [lib_file, exp_file],
        arguments = [
            "/d", "/c",
            "{} {} {} {}".format(
                script.path.replace("/", "\\"),
                dll_file.path.replace("/", "\\"),
                lib_file.path.replace("/", "\\"),
                exp_file.path.replace("/", "\\"),
            ),
        ],
        use_default_shell_env = True,
        mnemonic = "DllToImplib",
        progress_message = "Generating import lib %s" % lib_file.short_path,
    )
    return lib_file

def _copy(ctx, src_file, dst_path):
    # TODO: Use extra toolchain
    dst_file = ctx.actions.declare_file(dst_path)
    if ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo]):
        ctx.actions.run(
            executable = "cmd.exe",
            inputs = [ src_file ],
            outputs = [ dst_file ],
            use_default_shell_env = True,
            arguments = [
                "/d",
                "/c",
                'copy /Y "{}" "{}"'.format(
                    src_file.path.replace("/", "\\"),
                    dst_file.path.replace("/", "\\"),
                ),
            ],
        )
    else:
        ctx.actions.run(
            executable = "cp",
            inputs = [ src_file ],
            outputs = [ dst_file ],
            use_default_shell_env = True,
            arguments = [ src_file.path, dst_file.path ],
        )
    return dst_file

def _copy_crlf(ctx, src_file, dst_path):
    """Copy `src_file` to `dst_path` converting line endings to CRLF.

    Make normalizes the files it stages through its `.release.x` recipe with
    `sed -n -z -e 's/\\r*\\n/\\r\\n/g;p'` when `OS_is_win`, so text files such as
    `config/config.txt` ship with CRLF even though they are stored with LF in
    the repository. Plain `copy` leaves them as LF and the released file then
    differs from the Make one on every line.
    """
    dst_file = ctx.actions.declare_file(dst_path)
    script = ctx.file._to_crlf
    ctx.actions.run(
        executable = "cmd.exe",
        inputs = [src_file, script],
        outputs = [dst_file],
        use_default_shell_env = True,
        arguments = [
            "/d",
            "/c",
            "{} {} {}".format(
                script.path.replace("/", "\\"),
                src_file.path.replace("/", "\\"),
                dst_file.path.replace("/", "\\"),
            ),
        ],
        mnemonic = "CopyCrlf",
        progress_message = "Staging %s with CRLF line endings" % dst_file.short_path,
    )
    return dst_file

def _try_relativize(path, start):
    if path.startswith(start):
        return paths.relativize(path, start)
    return path

def _copy_version_header(ctx, src_file, dst_path, version_info, is_windows):
    dst_file = ctx.actions.declare_file(dst_path)

    # Make rewrites these seven lines with `sed`, and on Windows every
    # replacement carries a trailing `\r` (`sed.eol.win` in
    # dev/make/common.mk:177, used by `update_headers_version` in
    # makefile.ver:62-80). The rest of the header keeps the LF endings it has in
    # the repository, so the released file is deliberately mixed: seven CRLF
    # lines among LF ones. Reproduce that instead of writing the whole file with
    # one ending, or the release comparison reports a 7-byte difference.
    eol = "\r" if is_windows else ""
    ctx.actions.expand_template(
        template = src_file,
        output = dst_file,
        substitutions = {
            "#define __INTEL_DAAL_BUILD_DATE 21990101": "#define __INTEL_DAAL_BUILD_DATE {}{}".format(version_info.build, eol),
            "#define __INTEL_DAAL__        2199": "#define __INTEL_DAAL__ {}{}".format(version_info.major, eol),
            "#define __INTEL_DAAL_MINOR__  9": "#define __INTEL_DAAL_MINOR__ {}{}".format(version_info.minor, eol),
            "#define __INTEL_DAAL_UPDATE__ 9": "#define __INTEL_DAAL_UPDATE__ {}{}".format(version_info.update, eol),
            "#define __INTEL_DAAL_STATUS__ 'A'": "#define __INTEL_DAAL_STATUS__ \"{}\"{}".format(version_info.status, eol),
            "#define __INTEL_DAAL_MAJOR_BINARY__ 999": "#define __INTEL_DAAL_MAJOR_BINARY__ {}{}".format(version_info.binary_major, eol),
            "#define __INTEL_DAAL_MINOR_BINARY__ 999": "#define __INTEL_DAAL_MINOR_BINARY__ {}{}".format(version_info.binary_minor, eol),
        },
    )
    return dst_file

def _strip_os_suffix(dst_path, os_suffix):
    """Drop a trailing `_<os>` from `dst_path`'s basename, if present.

    Mirrors Make's `$(subst _$(_OS),,$d)` when staging
    `release.HEADERS.OSSPEC`: `include/daal_win.h` ships as `include/daal.h`.
    """
    base = paths.basename(dst_path)
    stem, _, extension = base.rpartition(".")
    if not stem or not stem.endswith(os_suffix):
        return dst_path
    stripped = "{}.{}".format(stem[:-len(os_suffix)], extension)
    return paths.join(paths.dirname(dst_path), stripped)

def _copy_include(ctx, prefix, version_info):
    include_prefix = paths.join(prefix, "include")
    is_windows = ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )
    # Make derives this from `$(_OS)`; only `win` currently has OS-specific
    # public headers, but keep the other platforms symmetrical.
    os_suffix = "_win" if is_windows else "_lnx"

    # Map each staged destination to the header that should provide it. An
    # OS-specific header wins over the generic file of the same staged name,
    # matching Make's `filter-out $(subst _$(_OS),,...)` against
    # `release.HEADERS.COMMON`. Keeping one entry per destination also prevents
    # declaring the same action output twice.
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
                # Use short_path for deterministic workspace-relative include layout.
                # file.path contains bazel-out/<cfg>/... prefixes that prevent
                # _try_relativize(skip_prefix) from matching and may leak bazel-out
                # directories into release/include.
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
                                            is_windows)
        else:
            dst_file = _copy(ctx, header, dst_path)
        dst_files.append(dst_file)
    return dst_files

def _symlink(ctx, link_name, target_name, prefix):
    """Create a relative symlink in the release directory.

    Args:
        ctx: Rule context.
        link_name: Basename of the symlink to create (e.g. "libonedal_core.so").
        target_name: Basename of the symlink target (e.g. "libonedal_core.so.2").
        prefix: Directory prefix inside the rule's output tree.

    Returns:
        The declared symlink File object.
    """
    link_file = ctx.actions.declare_symlink(paths.join(prefix, link_name))
    ctx.actions.symlink(
        output = link_file,
        # Relative symlink — target lives in the same directory.
        target_path = target_name,
    )
    return link_file

def _copy_lib(ctx, prefix, version_info):
    """Copy libraries to release directory with versioning and symlinks for .so files.

    For each shared library (.so) on Linux, this creates:
      libonedal_core.so.{binary_major}.{binary_minor}   (real file)
      libonedal_core.so.{binary_major}  -> libonedal_core.so.{binary_major}.{binary_minor}  (symlink)
      libonedal_core.so                 -> libonedal_core.so.{binary_major}                 (symlink)

    Static libraries (.a) and Windows DLLs (.dll) are copied as-is without versioning.

    macOS .dylib versioning is not yet implemented; .dylib files are currently
    copied as-is like static libraries.
    """
    lib_prefix = paths.join(prefix, "lib", "intel64")
    redist_prefix = paths.join(prefix, "redist", "intel64")
    is_windows = ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo])
    libs = _collect_default_files(ctx.attr.lib)
    dst_files = []

    # On Windows, when the dynamic library rule already emitted a real
    # import library, use that via rename; otherwise derive it post-link
    # from the .dll via dumpbin + lib /def.
    #
    # rules_cc may expose `<name>.if.lib`, while our cc_dynamic_lib rule
    # publishes the same file as `<name>_dll.lib`. Both represent the DLL
    # import library and must be consumed by the DLL branch below, otherwise
    # the final loop would try to copy `<name>_dll.lib` into the same release
    # output that the DLL branch is also generating.
    import_lib_stems = {}
    if is_windows:
        for lib in libs:
            if lib.basename.endswith(".if.lib"):
                import_lib_stems[lib.basename[:-len(".if.lib")]] = lib
            elif lib.basename.endswith("_dll.lib"):
                import_lib_stems[lib.basename[:-len("_dll.lib")]] = lib

    for lib in libs:
        # Determine if this is a shared library that needs versioning.
        # Only Linux .so files are versioned; .dylib (macOS) is not yet supported
        # and will be copied as-is like static libraries.
        is_shared_lib = lib.extension == "so"

        if is_shared_lib and not version_info:
            fail(("Shared library '{}' requires VersionInfo for SONAME versioning, " +
                 "but no version_info was provided to _copy_lib.").format(lib.basename))

        if is_shared_lib and version_info:
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
        else:
            # Static libs (.a), DLLs (.dll), import libs (.lib).
            # Windows release layout: place DAAL runtime DLLs under
            # redist/intel64 (not lib/intel64) and derive each DLL's
            # import library `<name>_dll.lib` via dumpbin+lib /def — the
            # makefile route (`-IMPLIB:<name>_dll.lib`) is unreachable
            # through Bazel's cc_common.link() so we generate the import
            # lib post-link, matching the Make release contents.
            if is_windows and lib.extension == "dll":
                dst_files.append(_copy(ctx, lib, paths.join(redist_prefix, lib.basename)))

                # `onedal_core.4.dll` -> `onedal_core_dll.lib`
                base_no_ver = lib.basename
                if version_info:
                    suffix = ".{}.dll".format(version_info.binary_major)
                    if base_no_ver.endswith(suffix):
                        base_no_ver = base_no_ver[:-len(suffix)] + ".dll"
                stem = base_no_ver[:-len(".dll")]
                # The thread import library is not shipped in the Make
                # DAAL package, mirror that here.
                if stem in _THREAD_STEMS:
                    continue
                implib_name = "{}_dll.lib".format(stem)
                # Prefer the link-emitted .if.lib (rules_cc MSVC auto-
                # config writes one alongside every DLL). Fall back to
                # dumpbin + lib /def: when the toolchain did not emit
                # one (icx custom config currently does not).
                source_import_lib = import_lib_stems.get(stem)
                if source_import_lib:
                    implib = _copy(
                        ctx, source_import_lib,
                        paths.join(lib_prefix, implib_name),
                    )
                else:
                    implib = _make_implib_for_dll(
                        ctx, lib, paths.join(lib_prefix, implib_name),
                    )
                dst_files.append(implib)
                if version_info and stem in _VERSIONED_IMPLIB_STEMS:
                    versioned_implib_name = "{}_dll.{}.lib".format(
                        stem,
                        version_info.binary_major,
                    )
                    dst_files.append(_copy(
                        ctx, implib,
                        paths.join(lib_prefix, versioned_implib_name),
                    ))
                continue
            # Link-emitted import libraries are consumed via the dll branch
            # above (renamed to `_dll.lib`); skip them here so we do not
            # declare the same output twice.
            if is_windows and (lib.basename.endswith(".if.lib") or lib.basename.endswith("_dll.lib")):
                continue
            if is_windows and lib.extension == "exp":
                continue

            dst_path = paths.join(lib_prefix, lib.basename)
            dst_files.append(_copy(ctx, lib, dst_path))

    return dst_files

# Files the Make Windows package ships with CRLF, which Bazel therefore has to
# convert as well or the released file differs on every line. Two independent
# mechanisms produce them:
#
#   * Make's `.release.x` recipe pipes whatever it stages through
#     `sed -n -z -e 's/\r*\n/\r\n/g;p'` when `OS_is_win` (makefile:1043). That
#     covers `config/config.txt` (makefile:1049) and the dataset tree
#     (makefile:1046); both are stored with LF in the repository
#     (see `.gitattributes`).
#   * cmake's `configure_file` normalises output to the *host's* newline rather
#     than the input's, so the oneDALConfig files that the makefile stages via
#     `cmake/scripts/generate_config.cmake` (makefile:1103) come out CRLF on
#     Windows even though the templates are LF. Bazel writes them with
#     `expand_template`, which keeps the template's LF.
#
# The reference direction is deliberate: Make's bytes are what has shipped in
# every release, and `generate_config.cmake` is also called by
# `deploy/nuget/prepare_dal_nuget.sh`, so teaching it `NEWLINE_STYLE LF` would
# change the published NuGet packages to fix a comparison.
#
# `env/vars.bat` is deliberately absent: it is generated from a template already
# checked in with CRLF, so both sides agree without help.
_CRLF_EXTRA_FILES = [
    "config/config.txt",
    "lib/cmake/oneDAL/oneDALConfig.cmake",
    "lib/cmake/oneDAL/oneDALConfigVersion.cmake",
]

# Make picks what to ship under `data/` with `expat` (makefile:395); mirror that
# suffix list rather than converting whatever happens to live there.
#
# Examples and samples are *not* covered: they are staged through the earlier
# `.release.x` and `.release.d` definitions (makefile:1026, :1053), neither of
# which runs the line-ending sed, so they keep LF in both packages.
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
    """Copy extra generated files (vars.sh, pkg-config, etc.) into the release tree.

    Each entry in extra_files is a (label, dst_subpath) pair encoded as two
    parallel lists: extra_files and extra_files_dst.

    dst_subpath is the desired path *relative to the release root*
    (e.g. "env/vars.sh", "lib/pkgconfig/onedal.pc").
    """
    if len(ctx.attr.extra_files) != len(ctx.attr.extra_files_dst):
        fail("extra_files and extra_files_dst must have the same length: got {} vs {}".format(
            len(ctx.attr.extra_files), len(ctx.attr.extra_files_dst)))

    is_windows = ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo])

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
        # See `_is_crlf_staged`: some of these are CRLF in the Make Windows
        # package, either because Make's staging recipe rewrites them or because
        # cmake generated them on a Windows host. The pkg-config files are not,
        # since Bazel and Make both write them with LF.
        if is_windows and _is_crlf_staged(dst_subpath):
            dst_files.append(_copy_crlf(ctx, src, dst_path))
        else:
            dst_files.append(_copy(ctx, src, dst_path))
    return dst_files

def _copy_data(ctx, prefix):
    """Copy data files (datasets, examples, config) preserving directory structure."""
    is_windows = ctx.target_platform_has_constraint(ctx.attr._windows_constraint[platform_common.ConstraintValueInfo])

    dst_files = []
    for dep in ctx.attr.data:
        srcs = dep[DefaultInfo].files.to_list()
        for src in srcs:
            # Skip external repo files (e.g., ../mkl_repo...)
            if src.short_path.startswith("../"):
                continue
            dst_path = paths.join(prefix, src.short_path)
            if is_windows and _is_crlf_staged(src.short_path):
                dst_files.append(_copy_crlf(ctx, src, dst_path))
            else:
                dst_files.append(_copy(ctx, src, dst_path))
    return dst_files

def _copy_to_release_impl(ctx):
    extra_toolchain = ctx.toolchains["@onedal//dev/bazel/toolchains:extra"]
    prefix = ctx.attr.name + "/daal/latest"
    version_info = ctx.attr._version_info[VersionInfo] if ctx.attr._version_info else None
    files = []
    files += _copy_include(ctx, prefix, version_info)
    files += _copy_lib(ctx, prefix, version_info)
    files += _copy_extra_files(ctx, prefix)
    files += _copy_data(ctx, prefix)
    return [DefaultInfo(files=depset(files))]

def _release_cpu_all_transition_impl(settings, attr):
    """Transition @config//:cpu to 'all' when it is the default ('auto').
    This ensures `bazel build //:release` compiles all ISA variants by default.
    Explicit non-auto overrides (e.g. --cpu=avx2) are respected.
    Note: Bazel cannot distinguish between an unset --cpu and explicit --cpu=auto;
    both are overridden to 'all'.
    """
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
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
        ),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_dll_to_implib": attr.label(
            default = "@onedal//dev/bazel/toolchains/tools:dll_to_implib.bat",
            allow_single_file = True,
            doc = "Helper that derives a Windows DLL's import library by " +
                  "running dumpbin+lib /def: post-link.",
        ),
        "_to_crlf": attr.label(
            default = "@onedal//dev/bazel/toolchains/tools:copy_crlf.bat",
            allow_single_file = True,
            doc = "Helper that copies a text file converting line endings to " +
                  "CRLF, matching the makefile's Windows release staging.",
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
    toolchains = [
        "@onedal//dev/bazel/toolchains:extra"
    ],
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

_FEATURES_OPTION = "//command_line_option:features"
_MSVC_RUNTIME_SETTING = str(Label("@config//:msvc_runtime"))
_MSVC_RUNTIME_DEBUG_FEATURE = "msvc_runtime_debug"

# Subdirectories of the release tree whose contents depend on the MSVC
# runtime. Everything else (headers, datasets, examples, env scripts) is
# runtime-independent and is taken from the release-runtime build only, so
# the two trees can be merged without output collisions. Note this must stay
# narrower than `lib/`: `lib/pkgconfig` and `lib/cmake` hold identically
# named files in both flavours and are taken from the release build only.
_RUNTIME_SPECIFIC_DIRS = ["lib/intel64/", "redist/intel64/"]

def _force_msvc_runtime(settings, debug):
    """Pin the MSVC runtime configuration, ignoring any inherited setting.

    Both the `--features` flag (which drives `-MD`/`-MDd` in the toolchain)
    and the `@config//:msvc_runtime` build setting (which dependencies
    `select()` on) are rewritten together, so a `--config=mdd` on the command
    line cannot leave one half of the pair pointing the other way.
    """
    features = [
        f
        for f in settings[_FEATURES_OPTION]
        if f != _MSVC_RUNTIME_DEBUG_FEATURE and f != "-" + _MSVC_RUNTIME_DEBUG_FEATURE
    ]
    if debug:
        features = features + [_MSVC_RUNTIME_DEBUG_FEATURE]
    return {
        _FEATURES_OPTION: features,
        _MSVC_RUNTIME_SETTING: "debug" if debug else "release",
    }

def _force_msvc_runtime_release_impl(settings, attr):
    return _force_msvc_runtime(settings, debug = False)

def _force_msvc_runtime_debug_impl(settings, attr):
    return _force_msvc_runtime(settings, debug = True)

_force_msvc_runtime_release_transition = transition(
    implementation = _force_msvc_runtime_release_impl,
    inputs = [_FEATURES_OPTION],
    outputs = [_FEATURES_OPTION, _MSVC_RUNTIME_SETTING],
)

_force_msvc_runtime_debug_transition = transition(
    implementation = _force_msvc_runtime_debug_impl,
    inputs = [_FEATURES_OPTION],
    outputs = [_FEATURES_OPTION, _MSVC_RUNTIME_SETTING],
)

def _relativize_release_file(file, dep):
    """Return `file`'s path relative to `dep`'s release root, or None.

    The release rule declares its outputs under `<target name>/daal/latest`
    within its own package, so a generated file's `short_path` ends up as
    `[<package>/]release/daal/latest/lib/intel64/onedal.lib`. Locate the
    release root by its `<name>/daal/latest` marker rather than assuming the
    target sits in the root package.
    """
    marker = paths.join(dep.label.name, "daal", "latest") + "/"
    index = file.short_path.find(marker)
    if index == -1:
        return None
    return file.short_path[index + len(marker):]

def _copy_from_release_tree(ctx, dep, prefix, only_dirs = None):
    dst_files = []
    for src in dep[DefaultInfo].files.to_list():
        rel = _relativize_release_file(src, dep)
        if rel == None:
            fail("Unexpected file '{}' in release tree of {}".format(
                src.short_path, dep.label))
        if only_dirs != None:
            if not [d for d in only_dirs if rel.startswith(d)]:
                continue
        dst_files.append(_copy(ctx, src, paths.join(prefix, rel)))
    return dst_files

def _get_single_dep(deps, attr_name):
    """Return the one configured target of a transitioned label_list attribute.

    `release_all` passes exactly one label per attribute; using label_list
    (rather than a plain label) keeps the value shape identical to the
    existing `_release_cpu_all_transition` attributes in this file, which are
    also label_lists holding ordinary configured targets.
    """
    if len(deps) != 1:
        fail("Attribute '{}' must hold exactly one target, got {}".format(
            attr_name, len(deps)))
    return deps[0]

def _release_all_impl(ctx):
    release_md = _get_single_dep(ctx.attr.release_md, "release_md")
    is_windows = ctx.target_platform_has_constraint(
        ctx.attr._windows_constraint[platform_common.ConstraintValueInfo],
    )
    if not is_windows:
        # The MSVC runtime distinction does not exist here, so there is
        # nothing to merge. Forward the single release tree as-is rather than
        # copying it: the Linux tree contains `.so` version symlinks, and
        # copying would dereference them into duplicate real files.
        return [DefaultInfo(files = release_md[DefaultInfo].files)]
    prefix = ctx.attr.name + "/daal/latest"
    files = _copy_from_release_tree(ctx, release_md, prefix)
    # Only the libraries differ between the two runtimes, and they carry the
    # `d` suffix, so both flavours coexist in one lib/redist directory. The
    # pkg-config files describe the release runtime only; debug-runtime
    # consumers get the right names from oneDALConfig.cmake, which appends
    # its own DAL_DEBUG_SUFFIX.
    files += _copy_from_release_tree(
        ctx, _get_single_dep(ctx.attr.release_mdd, "release_mdd"), prefix,
        only_dirs = _RUNTIME_SPECIFIC_DIRS,
    )
    return [DefaultInfo(files = depset(files))]

_release_all = rule(
    implementation = _release_all_impl,
    attrs = {
        # label_list rather than label: matches the existing transitioned
        # attributes in this file and keeps the configured-target value shape
        # unambiguous. Exactly one entry is expected (see _get_single_dep).
        "release_md": attr.label_list(
            mandatory = True,
            cfg = _force_msvc_runtime_release_transition,
            doc = "Release tree built against the release MSVC runtime.",
        ),
        "release_mdd": attr.label_list(
            mandatory = True,
            cfg = _force_msvc_runtime_debug_transition,
            doc = "Release tree built against the debug MSVC runtime. " +
                  "Only its lib/intel64 and redist/intel64 contents are " +
                  "used; ignored entirely on non-Windows platforms.",
        ),
        "_windows_constraint": attr.label(
            default = "@platforms//os:windows",
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)

def release_all(name, release_target):
    """Assemble one release tree holding both Windows MSVC runtime flavours.

    Builds `release_target` twice — once against the release CRT (`-MD`) and
    once against the debug CRT (`-MDd`) — and merges the results. The debug
    libraries carry a `d` suffix (`onedal_cored.lib`), mirroring the Makefile,
    so both sets live side by side in `lib/intel64` and `redist/intel64`.

    On non-Windows platforms the runtime distinction does not exist and the
    output is equivalent to `release_target` alone.

    Args:
        name:           Target name (also the output directory prefix).
        release_target: The `release()` target to build in both flavours.
    """
    _release_all(
        name = name,
        release_md = [release_target],
        release_mdd = [release_target],
    )

def release_include(hdrs, skip_prefix="", add_prefix=""):
    return (hdrs, add_prefix, skip_prefix)

def release(name, include, lib, extra_files = [], data = []):
    """Assemble the oneDAL release directory tree.

    Args:
        name:        Target name (also used as the output directory prefix).
        include:     List of release_include() tuples for headers.
        lib:         List of library targets (static, shared).
        extra_files: List of (label, dst_path) tuples for additional files.
                     Use release_extra_file() helper to construct entries.
                     Example:
                       extra_files = [
                           release_extra_file(":release_vars_sh", "env/vars.sh"),
                           release_extra_file(":release_pkgconfig", "lib/pkgconfig/onedal.pc", windows_dst_path = ""),
                       ]
        data:        List of filegroup targets to copy preserving directory structure.
                     Example:
                       data = [
                           "//data:datasets",
                           "//deploy/local:config",
                       ]
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
    )

def release_extra_file(label, dst_path, windows_dst_path = None):
    """Helper to declare an extra file for release().

    Args:
        label:    Bazel label of the target producing the file.
        dst_path: Destination path relative to the release root (e.g. "env/vars.sh").

    Returns:
        A tuple (label, dst_path) for use in release(extra_files=...).
    """
    return (label, dst_path, dst_path if windows_dst_path == None else windows_dst_path)
