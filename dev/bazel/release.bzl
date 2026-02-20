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

def _match_file_name(file, entries):
    for entry in entries:
        if entry in file.path:
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
    # TODO: Use extra toolchain
    dst_file = ctx.actions.declare_file(dst_path)
    ctx.actions.run(
        executable = "cp",
        inputs = [ src_file ],
        outputs = [ dst_file ],
        use_default_shell_env = True,
        arguments = [ src_file.path, dst_file.path ],
    )
    return dst_file

def _try_relativize(path, start):
    if path.startswith(start):
        return paths.relativize(path, start)
    return path

def _copy_include(ctx, prefix):
    include_prefix = paths.join(prefix, "include")
    dst_files = []
    for include, prefix, skip_prefix in zip(ctx.attr.include, ctx.attr.include_prefix,
                                            ctx.attr.include_skip_prefix):
        headers = _collect_headers(include)
        for header in headers:
            if skip_prefix:
                dst_path = _try_relativize(header.path, skip_prefix)
            elif prefix:
                dst_path = paths.join(prefix, header.basename)
            dst_file = _copy(ctx, header, paths.join(include_prefix, dst_path))
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

    For each shared library (.so / .dylib) on Linux/macOS, this creates:
      libonedal_core.so.{binary_major}.{binary_minor}   (real file)
      libonedal_core.so.{binary_major}  -> .so.{major}.{minor}  (symlink)
      libonedal_core.so                 -> .so.{major}           (symlink)

    Static libraries and Windows DLLs are copied as-is without versioning.
    """
    lib_prefix = paths.join(prefix, "lib", "intel64")
    libs = _collect_default_files(ctx.attr.lib)
    dst_files = []

    for lib in libs:
        # Determine if this is a shared library that needs versioning.
        # On Linux: .so extension; on macOS: .dylib (handled separately).
        is_shared_lib = lib.extension == "so"

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
            # Static libs (.a), DLLs (.dll), import libs (.lib) — copy as-is.
            dst_path = paths.join(lib_prefix, lib.basename)
            dst_files.append(_copy(ctx, lib, dst_path))

    return dst_files

def _copy_extra_files(ctx, prefix):
    """Copy extra generated files (vars.sh, pkg-config, etc.) into the release tree.

    Each entry in extra_files is a (label, dst_subpath) pair encoded as two
    parallel lists: extra_files and extra_files_dst.

    dst_subpath is the desired path *relative to the release root*
    (e.g. "env/vars.sh", "lib/pkgconfig/onedal.pc").
    """
    dst_files = []
    for dep, dst_subpath in zip(ctx.attr.extra_files, ctx.attr.extra_files_dst):
        srcs = dep[DefaultInfo].files.to_list()
        if len(srcs) != 1:
            fail("extra_files entry '{}' must produce exactly one file, got {}".format(
                dep.label, len(srcs)))
        src = srcs[0]
        dst_path = paths.join(prefix, dst_subpath)
        dst_files.append(_copy(ctx, src, dst_path))
    return dst_files

def _copy_to_release_impl(ctx):
    extra_toolchain = ctx.toolchains["@onedal//dev/bazel/toolchains:extra"]
    prefix = ctx.attr.name + "/daal/latest"
    version_info = ctx.attr._version_info[VersionInfo] if ctx.attr._version_info else None
    files = []
    files += _copy_include(ctx, prefix)
    files += _copy_lib(ctx, prefix, version_info)
    files += _copy_extra_files(ctx, prefix)
    return [DefaultInfo(files=depset(files))]

_release = rule(
    implementation = _copy_to_release_impl,
    attrs = {
        "include": attr.label_list(allow_files=True),
        "include_prefix": attr.string_list(),
        "include_skip_prefix": attr.string_list(),
        "lib": attr.label_list(allow_files=True),
        "extra_files": attr.label_list(
            allow_files = True,
            doc = "Additional generated files to include in release. "
                  "Must be paired 1:1 with extra_files_dst.",
        ),
        "extra_files_dst": attr.string_list(
            doc = "Destination paths for extra_files, relative to the release root. "
                  "E.g. ['env/vars.sh', 'lib/pkgconfig/onedal.pc'].",
        ),
        "_version_info": attr.label(
            default = "@config//:version",
            providers = [VersionInfo],
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

def release_include(hdrs, skip_prefix="", add_prefix=""):
    return (hdrs, add_prefix, skip_prefix)

def release(name, include, lib, extra_files = []):
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
                           release_extra_file(":release_pkgconfig", "lib/pkgconfig/onedal.pc"),
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
    for label, dst in extra_files:
        rule_extra_files.append(label)
        rule_extra_files_dst.append(dst)

    _release(
        name = name,
        include = rule_include,
        include_prefix = rule_include_prefix,
        include_skip_prefix = rule_include_skip_prefix,
        lib = lib,
        extra_files = rule_extra_files,
        extra_files_dst = rule_extra_files_dst,
    )

def release_extra_file(label, dst_path):
    """Helper to declare an extra file for release().

    Args:
        label:    Bazel label of the target producing the file.
        dst_path: Destination path relative to the release root (e.g. "env/vars.sh").

    Returns:
        A tuple (label, dst_path) for use in release(extra_files=...).
    """
    return (label, dst_path)
