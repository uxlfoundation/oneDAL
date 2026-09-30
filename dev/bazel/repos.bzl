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

_BINARY_MAJOR = "4"
_BINARY_MINOR = "0"

def _download_and_extract(repo_ctx, url, sha256, output, strip_prefix):
    # Workaround Python wheel extraction. Bazel cannot determine file
    # type automatically as does not support wheels out-of-the-box.
    filename = url.split("/")[-1]
    downloaded_path = repo_ctx.path(filename)
    repo_ctx.download(
        url = url,
        output = downloaded_path,
        sha256 = sha256,
    )

    if filename.endswith(".conda"):
        repo_ctx.execute(["unzip", downloaded_path, "-d", output])
        for entry in repo_ctx.path(output).readdir():
            if entry.basename.startswith("pkg-") and entry.basename.endswith(".tar.zst"):
                repo_ctx.execute(["sh", "-c", "unzstd '%s' --stdout | tar -xf - -C '%s'" % (entry, output)])

    elif filename.endswith(".whl") or filename.endswith(".zip"):
        repo_ctx.download_and_extract(
            url = url,
            sha256 = sha256,
            output = output,
            strip_prefix = strip_prefix,
            type = "zip",
        )

    else:
        repo_ctx.download_and_extract(
            url = url,
            sha256 = sha256,
            output = output,
            strip_prefix = strip_prefix,
        )



def _detect_os(repo_ctx):
    return "win" if repo_ctx.os.name.lower().find("windows") != -1 else "lnx"

def _select_by_os(repo_ctx, name, os_id):
    if os_id == "win":
        win_name = ("_win" + name) if name.startswith("_") else ("win_" + name)
        if hasattr(repo_ctx.attr, win_name):
            win_value = getattr(repo_ctx.attr, win_name)
            if win_value:
                return win_value
    return getattr(repo_ctx.attr, name)

def _archive(url, sha256, strip_prefix = ""):
    """Describe one downloadable archive of a prebuilt dependency.

    The repository rule stores the coordinates as three parallel string lists,
    because that is the only list type a rule attribute can hold. Authors pass
    them as a list of these structs instead, so that a URL never drifts apart
    from its own hash and prefix when a dependency is bumped.

    Args:
        url: Location of the archive. `.conda`, `.whl` and `.zip` are unpacked
             by `_download_and_extract`; anything else is left to Bazel.
        sha256: Expected hash of the archive.
        strip_prefix: Directory prefix to drop while unpacking. Empty for
                      packages that already unpack into the required layout.

    Returns:
        A struct holding the three coordinates.
    """
    return struct(url = url, sha256 = sha256, strip_prefix = strip_prefix)

def _unzip_archives(archives):
    return struct(
        urls = [a.url for a in archives],
        sha256s = [a.sha256 for a in archives],
        strip_prefixes = [a.strip_prefix for a in archives],
    )

def _normalize_download_info(repo_ctx, os_id):
    urls = _select_by_os(repo_ctx, "urls", os_id)
    sha256s = _select_by_os(repo_ctx, "sha256s", os_id)
    strip_prefixes = _select_by_os(repo_ctx, "strip_prefixes", os_id)
    if len(sha256s) != len(urls):
        fail("sha256 hashes count does not match URLs count")
    if len(strip_prefixes) != len(urls):
        fail("strip_prefixes count does not match URLs count")
    return [
        struct(url = url, sha256 = sha256, strip_prefix = strip_prefix)
        for url, sha256, strip_prefix in zip(urls, sha256s, strip_prefixes)
    ]

def _split_glob_entry(root, entry_fmt, mapping):
    """Split a globbed entry into the directory to scan and the basename pattern."""
    pattern = entry_fmt.split("/")[-1]
    dir_part = entry_fmt[:entry_fmt.rfind("/")] if "/" in entry_fmt else ""
    root_with_dir = utils.substitute(
        paths.join(root, dir_part) if dir_part else root,
        mapping
    )
    return struct(pattern = pattern, dir_part = dir_part, root_with_dir = root_with_dir)

def _glob_dir_entries(repo_ctx, glob):
    """List the directory a globbed entry scans, or nothing if it is absent.

    `readdir()` on a path that does not exist is an error, not an empty list, so
    a pattern under a directory the package does not ship has to be answered
    before the directory is enumerated. Callers that require a match still
    report it themselves; the optional ones treat it as "not present", which is
    their contract.
    """
    dir_path = repo_ctx.path(glob.root_with_dir)
    if not dir_path.exists or not dir_path.is_dir:
        return []
    return dir_path.readdir()

def _glob_entry_matches(repo_ctx, root, entry_fmt, mapping):
    glob = _split_glob_entry(root, entry_fmt, mapping)
    for fs_entry in _glob_dir_entries(repo_ctx, glob):
        if _matches_glob(fs_entry.basename, glob.pattern):
            return True
    return False

def _create_symlinks(repo_ctx, root, entries, substitutions=None, mapping=None):
    substitutions = substitutions or {}
    mapping = mapping or {}

    for entry in entries:
        entry_fmt = utils.substitute(entry, substitutions)
        if "*" in entry_fmt:
            glob = _split_glob_entry(root, entry_fmt, mapping)
            matched = False
            for fs_entry in _glob_dir_entries(repo_ctx, glob):
                if _matches_glob(fs_entry.basename, glob.pattern):
                    matched = True
                    dst = (paths.join(glob.dir_part, fs_entry.basename)
                           if glob.dir_part else fs_entry.basename)
                    repo_ctx.symlink(str(fs_entry), dst)
            if not matched:
                fail("No files matched pattern '%s' in directory '%s' while creating symlinks for entry '%s'" %
                     (glob.pattern, glob.root_with_dir, entry_fmt))
        else:
            src_entry_path = utils.substitute(paths.join(root, entry_fmt), mapping)
            dst_entry_path = entry_fmt
            repo_ctx.symlink(src_entry_path, dst_entry_path)

def _create_optional_symlinks(repo_ctx, root, entries, substitutions=None, mapping=None):
    """Symlink the entries the package actually ships, skipping the rest.

    Used for libraries whose presence depends on how the package was built, such
    as the separate parameter libraries. Presence is read off the file system
    rather than out of package metadata, so the result cannot disagree with the
    package, and OS-specific naming needs no separate attribute: entries that do
    not exist for the host OS are simply not there.

    Globbed entries are kept when the pattern matches at least one file, and
    dropped otherwise. Testing `exists` on the raw entry would never be true for
    a pattern, so a globbed entry would always be dropped silently.
    """
    substitutions = substitutions or {}
    mapping = mapping or {}
    present = []
    for entry in entries:
        entry_fmt = utils.substitute(entry, substitutions)
        if "*" in entry_fmt:
            if _glob_entry_matches(repo_ctx, root, entry_fmt, mapping):
                present.append(entry)
            continue
        src_entry_path = utils.substitute(paths.join(root, entry_fmt), mapping)
        if repo_ctx.path(src_entry_path).exists:
            present.append(entry)
    _create_symlinks(repo_ctx, root, present, substitutions, mapping)

def _create_required_any_symlinks(repo_ctx, root, entries, what, substitutions=None, mapping=None):
    """Symlink whichever of `entries` the package ships, requiring at least one.

    For a set of files that are individually optional but collectively required:
    the MKL CPU dispatch kernels, where which ISA families a release ships
    changes between versions but a layout with the classic libraries and no
    kernel at all is broken -- `libmkl_core.so.2` is only the dispatcher, so the
    first classic-MKL call dies with `Cannot load libmkl_avx512.so.2 or
    libmkl_def.so.2`. Failing in the repository rule names the package; failing
    at run time names a kernel the user never asked for.
    """
    if not entries:
        return
    substitutions = substitutions or {}
    mapping = mapping or {}
    present = []
    for entry in entries:
        entry_fmt = utils.substitute(entry, substitutions)
        if "*" in entry_fmt:
            if _glob_entry_matches(repo_ctx, root, entry_fmt, mapping):
                present.append(entry)
            continue
        if repo_ctx.path(utils.substitute(paths.join(root, entry_fmt), mapping)).exists:
            present.append(entry)
    if not present:
        fail(("None of {} matched in '{}', and at least one is required: {}. " +
              "Point {} at a package that ships them.").format(
            entries, root, what, repo_ctx.name))
    _create_symlinks(repo_ctx, root, present, substitutions, mapping)

def _matches_glob(name, pattern):
    if "*" not in pattern:
        return name == pattern
    parts = pattern.split("*")
    if not name.startswith(parts[0]):
        return False
    if not name.endswith(parts[-1]):
        return False
    pos = len(parts[0])
    for part in parts[1:-1]:
        idx = name.find(part, pos)
        if idx == -1:
            return False
        pos = idx + len(part)
    return True

def _download(repo_ctx, os_id):
    output = repo_ctx.path("archive")
    info_entries = _normalize_download_info(repo_ctx, os_id)
    for info in info_entries:
        _download_and_extract(
            repo_ctx,
            url = info.url,
            sha256 = info.sha256,
            output = output,
            strip_prefix = info.strip_prefix,
        )
    return str(output)

def _prebuilt_libs_repo_impl(repo_ctx):
    os_id = _detect_os(repo_ctx)
    root = repo_ctx.os.environ.get(repo_ctx.attr.root_env_var)
    if root:
        mapping = {}
    else:
        if _select_by_os(repo_ctx, "urls", os_id):
            root = _download(repo_ctx, os_id)
            mapping = _select_by_os(repo_ctx, "_download_mapping", os_id)
        elif repo_ctx.attr.fallback_root:
            root = repo_ctx.attr.fallback_root
        else:
            fail("Cannot locate {} dependency".format(repo_ctx.name))
    substitutions = {
        "%{os}": os_id,
        "%{repo_root}": str(repo_ctx.path("")),
    }
    substitutions["%{version_binary_major}"] = _BINARY_MAJOR
    substitutions["%{version_binary_minor}"] = _BINARY_MINOR
    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "includes", os_id), substitutions, mapping)
    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "libs", os_id), substitutions, mapping)
    # `optional_libs` holds the libraries a package ships only in some layouts,
    # today the separate parameter libraries. Each is symlinked when the package
    # contains it. The BUILD template picks them up with
    # `glob(..., allow_empty = True)`, so an absent entry is what makes a folded
    # package resolve to a template without parameter libraries -- no template
    # substitution and no package metadata are involved.
    _create_optional_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "optional_libs", os_id), substitutions, mapping)
    # Read without `_select_by_os`, which falls back to the Linux value when the
    # `win_` one is empty. That is right for a list of files to symlink and wrong
    # for a requirement: Linux patterns can never match a Windows layout, so the
    # fallback would turn an empty Windows group into a guaranteed failure.
    _create_required_any_symlinks(
        repo_ctx, root,
        repo_ctx.attr.win_required_any_libs if os_id == "win" else repo_ctx.attr.required_any_libs,
        repo_ctx.attr.required_any_libs_description,
        substitutions, mapping,
    )
    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "bins", os_id), substitutions, mapping)
    repo_ctx.template(
        "BUILD",
        _select_by_os(repo_ctx, "build_template", os_id),
        substitutions = substitutions,
    )

def _prebuilt_libs_repo_rule(includes, libs, build_template, bins=[], optional_libs=[],
                             required_any_libs=[], required_any_libs_description="",
                             win_required_any_libs=[],
                             root_env_var="", fallback_root="", archives=[],
                             download_mapping={},
                             win_includes=[], win_libs=[], win_bins=[], win_build_template=None,
                             win_archives=[], win_download_mapping={}):
    """Declare a repository rule for a dependency shipped as prebuilt binaries.

    Everything that describes one dependency -- the environment variable that
    points at a local installation, the archives to fall back on, the file
    layout and the BUILD template -- is passed here as a default, so that the
    dependency is fully described by its own `dev/bazel/deps/<dep>.bzl` file and
    `MODULE.bazel` only has to name it.

    Args:
        includes: Include directories to symlink into the repository, relative
                  to the dependency root. Entries may contain `*` globs and
                  `%{...}` substitutions.
        libs: Libraries to symlink. Missing entries are an error.
        build_template: BUILD file template for the repository.
        bins: Executables and runtime libraries to symlink.
        optional_libs: Libraries to symlink only if the dependency ships them.
        required_any_libs: Libraries of which at least one must be present; every
                           present one is symlinked.
        required_any_libs_description: Names the `required_any_libs` set in the
                                       error raised when none is present.
        win_required_any_libs: Windows override for `required_any_libs`.
        root_env_var: Environment variable holding a local installation root.
                      When set in the environment it wins over `archives`.
        fallback_root: Root to use when neither the variable nor `archives` is
                       available.
        archives: `repos.archive()` entries to download when no local
                  installation is pointed to.
        download_mapping: Maps the layout the entries above are written in (LHS)
                          onto the layout the downloaded archives actually have
                          (RHS), e.g. `{"lib/intel64": "lib/"}`.
        win_includes: Windows override for `includes`.
        win_libs: Windows override for `libs`.
        win_bins: Windows override for `bins`.
        win_build_template: Windows override for `build_template`.
        win_archives: Windows override for `archives`.
        win_download_mapping: Windows override for `download_mapping`.

    Returns:
        The repository rule.
    """
    download_info = _unzip_archives(archives)
    win_download_info = _unzip_archives(win_archives)
    return repository_rule(
        implementation = _prebuilt_libs_repo_impl,
        environ = [
            root_env_var,
        ],
        local = True,
        configure = True,
        attrs = {
            "root_env_var": attr.string(default=root_env_var),
            "fallback_root": attr.string(default=fallback_root),
            "urls": attr.string_list(default=download_info.urls),
            "sha256s": attr.string_list(default=download_info.sha256s),
            "strip_prefixes": attr.string_list(default=download_info.strip_prefixes),
            "includes": attr.string_list(default=includes),
            "libs": attr.string_list(default=libs),
            "optional_libs": attr.string_list(default=optional_libs),
            "required_any_libs": attr.string_list(default=required_any_libs),
            "required_any_libs_description": attr.string(default=required_any_libs_description),
            "win_required_any_libs": attr.string_list(default=win_required_any_libs),
            "bins": attr.string_list(default=bins),
            "build_template": attr.label(allow_files=True,
                                         default=Label(build_template)),
            "win_includes": attr.string_list(default=win_includes),
            "win_libs": attr.string_list(default=win_libs),
            "win_bins": attr.string_list(default=win_bins),
            "win_build_template": attr.label(allow_files=True,
                                             default=Label(win_build_template or build_template)),
            "win_urls": attr.string_list(default=win_download_info.urls),
            "win_sha256s": attr.string_list(default=win_download_info.sha256s),
            "win_strip_prefixes": attr.string_list(default=win_download_info.strip_prefixes),
            "_download_mapping": attr.string_dict(default=download_mapping),
            "_win_download_mapping": attr.string_dict(default=win_download_mapping),
        }
    )

repos = struct(
    archive = _archive,
    prebuilt_libs_repo_rule = _prebuilt_libs_repo_rule,
    prebuilt_libs_repo_impl = _prebuilt_libs_repo_impl,
    create_symlinks = _create_symlinks,
)
