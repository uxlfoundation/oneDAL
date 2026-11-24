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
load("@onedal//dev/bazel:version.bzl", "VERSION")
load("@onedal//dev/bazel/toolchains:common.bzl",
    "arch_dir",
    "detect_host_arch",
    "detect_os",
    "detect_target_arch",
)

def _extract_conda(repo_ctx, archive, output):
    """Unpacks a `.conda` package, a zip of `pkg-*.tar.zst` (the files) and `info-*.tar.zst`.

    Args:
        repo_ctx: repository context.
        archive: path of the downloaded `.conda` file.
        output: directory to unpack the package files into.
    """
    staging = repo_ctx.path(str(archive) + ".d")
    repo_ctx.extract(archive = archive, output = staging, type = "zip")
    for entry in staging.readdir():
        if entry.basename.startswith("pkg-") and entry.basename.endswith(".tar.zst"):
            repo_ctx.extract(archive = entry, output = output, type = "tar.zst")
    repo_ctx.delete(staging)

def _download_and_extract(repo_ctx, url, sha256, output, strip_prefix):
    archive = repo_ctx.path(url.split("/")[-1])
    repo_ctx.download(url = url, output = archive, sha256 = sha256)
    if archive.basename.endswith(".conda"):
        _extract_conda(repo_ctx, archive, output)
    else:
        # Bazel does not recognize a wheel as an archive by itself.
        archive_type = "zip" if archive.basename.endswith(".whl") else ""
        repo_ctx.extract(archive = archive, output = output, strip_prefix = strip_prefix, type = archive_type)
    repo_ctx.delete(archive)

def _detect_os(repo_ctx):
    # Every non-Windows host, macOS included, takes the Linux layout and templates.
    return "win" if detect_os(repo_ctx) == "win" else "lnx"

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

    Args:
        url: location of the archive; `.conda`, `.whl` and `.zip` are unpacked here, the rest by Bazel.
        sha256: expected hash of the archive.
        strip_prefix: directory prefix to drop while unpacking.
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

def _source_path(root, entry_fmt, mapping):
    """Path of `entry_fmt` under `root`, with the `mapping` onto the package layout applied."""
    return paths.join(root, utils.substitute(entry_fmt, mapping))

def _split_glob_entry(root, entry_fmt, mapping):
    """Split a globbed entry into the directory to scan and the basename pattern."""
    pattern = entry_fmt.split("/")[-1]
    dir_part = entry_fmt[:entry_fmt.rfind("/")] if "/" in entry_fmt else ""
    root_with_dir = _source_path(root, dir_part, mapping) if dir_part else root
    return struct(pattern = pattern, dir_part = dir_part, root_with_dir = root_with_dir)

def _glob_dir_entries(repo_ctx, glob):
    """List the directory a globbed entry scans, or nothing if it is absent (`readdir()` fails)."""
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

def _create_symlinks(repo_ctx, root, entries, substitutions=None, mapping=None, source=None):
    """Symlink every entry from `root` into the repository; a missing one fails the fetch.

    Args:
        repo_ctx: repository context.
        root: directory the entries are taken from.
        entries: paths relative to `root`; `*` globs a directory, `%{...}` takes `substitutions`.
        substitutions: `%{...}` placeholders and their values.
        mapping: entry paths and the paths they have in this package's layout.
        source: where `root` comes from, named in the error of a missing entry.
    """
    substitutions = substitutions or {}
    mapping = mapping or {}
    source = source or "'{}'".format(root)

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
                fail("{}: nothing matches '{}' in '{}' ({})".format(
                    repo_ctx.name, glob.pattern, glob.root_with_dir, source))
        else:
            src_entry_path = _source_path(root, entry_fmt, mapping)
            if not repo_ctx.path(src_entry_path).exists:
                fail("{}: '{}' is missing, '{}' does not exist ({})".format(
                    repo_ctx.name, entry_fmt, src_entry_path, source))
            repo_ctx.symlink(src_entry_path, entry_fmt)

def _present_entries(repo_ctx, root, entries, substitutions, mapping):
    """The entries the package ships; a globbed one when it matches a file."""
    present = []
    for entry in entries:
        entry_fmt = utils.substitute(entry, substitutions)
        if "*" in entry_fmt:
            if _glob_entry_matches(repo_ctx, root, entry_fmt, mapping):
                present.append(entry)
        elif repo_ctx.path(_source_path(root, entry_fmt, mapping)).exists:
            present.append(entry)
    return present

def _create_optional_symlinks(repo_ctx, root, entries, substitutions, mapping, source):
    """Symlink the entries the package ships and skip the rest."""
    present = _present_entries(repo_ctx, root, entries, substitutions, mapping)
    _create_symlinks(repo_ctx, root, present, substitutions, mapping, source)

def _create_required_any_symlinks(repo_ctx, root, entries, what, substitutions, mapping, source):
    """Symlink whichever of `entries` the package ships, requiring at least one."""
    if not entries:
        return
    present = _present_entries(repo_ctx, root, entries, substitutions, mapping)
    if not present:
        fail("{}: none of {} is in '{}', and at least one is required: {} ({})".format(
            repo_ctx.name, entries, root, what, source))
    _create_symlinks(repo_ctx, root, present, substitutions, mapping, source)

def _pick_mapping(repo_ctx, root, entries, substitutions, mappings):
    """The first of `mappings` under which every entry is present, else the first mapping."""
    for mapping in mappings:
        if len(_present_entries(repo_ctx, root, entries, substitutions, mapping)) == len(entries):
            return mapping
    return mappings[0]

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

def _normalize_pypi_name(name):
    return name.lower().replace("_", "-").replace(".", "-")

def _parse_pypi_lock(content):
    """Parses a hash-pinned requirements file into `{name: struct(version, sha256s)}`.

    Only the subset Renovate's `pip_requirements` manager (through `hashin`)
    writes is understood: one `name==version` per logical line, followed by
    `--hash=sha256:...` options, with `\\` line continuations and `#` comments.
    """
    packages = {}
    for line in content.replace("\\\n", " ").split("\n"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        tokens = [t for t in line.replace("\t", " ").split(" ") if t]
        if "==" not in tokens[0]:
            fail("pypi lock: expected `name==version`, got '{}'".format(tokens[0]))
        name, version = tokens[0].split("==")
        sha256s = [
            t[len("--hash=sha256:"):]
            for t in tokens[1:]
            if t.startswith("--hash=sha256:")
        ]
        if not sha256s:
            fail("pypi lock: '{}' has no --hash=sha256 entries".format(tokens[0]))
        packages[_normalize_pypi_name(name)] = struct(version = version, sha256s = sha256s)
    return packages

# Wheel platform tag that each OS downloads. Only x86_64 wheels are used, which
# `_check_download_arch` enforces.
_PYPI_PLATFORM_TAG = {
    "lnx": "manylinux_2_28_x86_64",
    "win": "win_amd64",
}

def _wheel_has_platform(repo_ctx, wheel_path, dist, version, platform_tag):
    """Tells whether a downloaded wheel declares `platform_tag` in its WHEEL file."""
    staging = repo_ctx.path(str(wheel_path) + ".d")
    repo_ctx.extract(archive = wheel_path, output = staging, type = "zip")
    wheel_file = staging.get_child("{}-{}.dist-info".format(dist, version)).get_child("WHEEL")
    found = wheel_file.exists and any([
        line.startswith("Tag:") and line.strip().endswith("-" + platform_tag)
        for line in repo_ctx.read(wheel_file).split("\n")
    ])
    repo_ctx.delete(staging)
    return found

def _fetch_pypi_wheel(repo_ctx, name, pin, platform_tag):
    """Downloads the wheel of `name` for `platform_tag`; returns its path and dist name.

    The lock does not say which hash is which platform's wheel, so the PyPI metadata picks it.
    Offline, each locked hash is tried and the wheel that declares `platform_tag` is kept.
    """
    metadata_path = repo_ctx.path("pypi/{}-{}.json".format(name, pin.version))
    metadata = repo_ctx.download(
        url = "https://pypi.org/pypi/{}/{}/json".format(name, pin.version),
        output = metadata_path,
        allow_fail = True,
    )
    if metadata.success:
        candidates = [
            f
            for f in json.decode(repo_ctx.read(metadata_path))["urls"]
            if f["filename"].endswith("-{}.whl".format(platform_tag))
        ]
        if len(candidates) != 1:
            fail("pypi package {}=={}: expected one '*-{}.whl' wheel, found {}".format(
                name, pin.version, platform_tag, [f["filename"] for f in candidates]))
        wheel = candidates[0]
        sha256 = wheel["digests"]["sha256"]
        if sha256 not in pin.sha256s:
            fail("pypi package {}=={}: the hash of {} ({}) is not in the lock file".format(
                name, pin.version, wheel["filename"], sha256))
        wheel_path = repo_ctx.path("pypi/" + wheel["filename"])
        repo_ctx.download(url = wheel["url"], output = wheel_path, sha256 = sha256)
        return wheel_path, wheel["filename"].split("-")[0]

    dist = name.replace("-", "_")
    urls = [
        "https://files.pythonhosted.org/packages/{tag}/{initial}/{name}/{dist}-{version}-{tag}-none-{platform}.whl".format(
            tag = python_tag, initial = name[0], name = name, dist = dist,
            version = pin.version, platform = platform_tag,
        )
        for python_tag in ["py2.py3", "py3"]
    ]
    for sha256 in pin.sha256s:
        wheel_path = repo_ctx.path("pypi/{}-{}-{}.whl".format(dist, pin.version, sha256[:12]))
        if not repo_ctx.download(url = urls, output = wheel_path, sha256 = sha256, allow_fail = True).success:
            continue
        if _wheel_has_platform(repo_ctx, wheel_path, dist, pin.version, platform_tag):
            return wheel_path, dist
        repo_ctx.delete(wheel_path)
    fail(("pypi package {}=={}: no locked {} wheel could be fetched; the PyPI " +
          "metadata was unreachable and no locked hash is in the repository cache").format(
        name, pin.version, platform_tag))

def _download_pypi(repo_ctx, os_id, package, lock, output):
    """Downloads and unpacks the wheel of `package` pinned in the lock file."""
    name = _normalize_pypi_name(package)
    if name not in lock:
        fail("pypi package '{}' is not pinned in {}".format(package, repo_ctx.attr._pypi_lock))
    pin = lock[name]
    wheel_path, dist = _fetch_pypi_wheel(repo_ctx, name, pin, _PYPI_PLATFORM_TAG[os_id])

    # Wheels of native libraries put their payload under `<dist>-<version>.data/data`,
    # and the Windows ones one level deeper, in a conda-style `Library` prefix.
    strip_prefix = "{}-{}.data/data".format(dist, pin.version)
    if os_id == "win":
        strip_prefix += "/Library"
    repo_ctx.extract(archive = wheel_path, output = output, strip_prefix = strip_prefix, type = "zip")
    repo_ctx.delete(wheel_path)

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
    pypi_packages = _select_by_os(repo_ctx, "pypi_packages", os_id)
    if pypi_packages:
        lock = _parse_pypi_lock(repo_ctx.read(repo_ctx.attr._pypi_lock))
        wheel_os = "win" if os_id == "win" and repo_ctx.attr.win_pypi_packages else "lnx"
        for package in pypi_packages:
            _download_pypi(repo_ctx, wheel_os, package, lock, output)
    return str(output)

def _check_download_arch(repo_ctx, os_id):
    """Fails instead of downloading a package built for an architecture other than the target's."""
    download_arch = _select_by_os(repo_ctx, "download_arch", os_id)
    target_arch = detect_target_arch(repo_ctx, detect_host_arch(repo_ctx, os_id))
    if download_arch and download_arch != target_arch:
        fail(("Cannot supply the {} dependency by download for this target: the " +
              "configured package is built for {}, the target is {}. Build it " +
              "from source and point {} at the result.").format(
            repo_ctx.name, download_arch, target_arch,
            repo_ctx.attr.root_env_var or "the dependency root env var",
        ))

def _release_arch_dir(repo_ctx, os_id):
    return arch_dir(os_id, detect_target_arch(repo_ctx, detect_host_arch(repo_ctx, os_id)))

def _prebuilt_libs_repo_impl(repo_ctx):
    os_id = _detect_os(repo_ctx)
    substitutions = {
        "%{os}": os_id,
        "%{repo_root}": str(repo_ctx.path("")),
        "%{arch_dir}": _release_arch_dir(repo_ctx, os_id),
        "%{version_binary_major}": VERSION.binary_major,
        "%{version_binary_minor}": VERSION.binary_minor,
    }
    libs = _select_by_os(repo_ctx, "libs", os_id)
    root = repo_ctx.os.environ.get(repo_ctx.attr.root_env_var)
    if root:
        source = "${}".format(repo_ctx.attr.root_env_var)
        fallback = repo_ctx.attr._fallback_mapping
        mapping = _pick_mapping(repo_ctx, root, libs, substitutions, [{}, fallback] if fallback else [{}])
    elif _select_by_os(repo_ctx, "urls", os_id) or _select_by_os(repo_ctx, "pypi_packages", os_id):
        _check_download_arch(repo_ctx, os_id)
        source = "the downloaded package"
        root = _download(repo_ctx, os_id)
        mapping = _select_by_os(repo_ctx, "_download_mapping", os_id)
    else:
        fail("Cannot locate {} dependency: neither ${} is set nor archives or PyPI packages are declared".format(
            repo_ctx.name,
            repo_ctx.attr.root_env_var,
        ))
    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "includes", os_id), substitutions, mapping, source)
    _create_symlinks(repo_ctx, root, libs, substitutions, mapping, source)
    # The BUILD templates glob these with `allow_empty = True`, so an absent library drops out.
    _create_optional_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "optional_libs", os_id), substitutions, mapping, source)
    # Not `_select_by_os`: its Linux fallback for an empty `win_` list could never match.
    _create_required_any_symlinks(
        repo_ctx, root,
        repo_ctx.attr.win_required_any_libs if os_id == "win" else repo_ctx.attr.required_any_libs,
        repo_ctx.attr.required_any_libs_description,
        substitutions, mapping, source,
    )
    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "bins", os_id), substitutions, mapping, source)
    repo_ctx.template(
        "BUILD",
        _select_by_os(repo_ctx, "build_template", os_id),
        substitutions = substitutions,
    )

def _prebuilt_libs_repo_rule(includes, libs, build_template, bins=[], optional_libs=[],
                             required_any_libs=[], required_any_libs_description="",
                             win_required_any_libs=[],
                             root_env_var="", archives=[], pypi_packages=[],
                             download_arch="intel64", win_download_arch="",
                             download_mapping={}, fallback_mapping={},
                             win_includes=[], win_libs=[], win_bins=[], win_optional_libs=[],
                             win_build_template=None,
                             win_archives=[], win_pypi_packages=[], win_download_mapping={}):
    """Declare a repository rule for a dependency shipped as prebuilt binaries.

    Args:
        includes: include directories to symlink, relative to the root; `*` and `%{...}` allowed.
        libs: libraries to symlink; a missing one fails the fetch.
        build_template: BUILD file template for the repository.
        bins: executables and runtime libraries to symlink.
        optional_libs: libraries to symlink only if the dependency ships them.
        required_any_libs: libraries of which at least one must be present.
        required_any_libs_description: names that set in the error when none is present.
        win_required_any_libs: Windows override for `required_any_libs`.
        root_env_var: variable holding a local installation root; wins over the downloads.
        archives: `repos.archive()` entries to download without a local installation.
        pypi_packages: PyPI packages to download at the versions in requirements_lock.txt.
        download_arch: oneDAL arch ID the downloads are built for; empty disables the check.
        win_download_arch: Windows override for `download_arch`.
        download_mapping: maps the entries' layout onto the archive's, e.g. `{"lib/intel64": "lib/"}`.
        fallback_mapping: second layout of a local installation, used when `libs` are not at the first.
        win_includes: Windows override for `includes`.
        win_libs: Windows override for `libs`.
        win_bins: Windows override for `bins`.
        win_optional_libs: Windows override for `optional_libs`.
        win_build_template: Windows override for `build_template`.
        win_archives: Windows override for `archives`.
        win_pypi_packages: Windows wheels; without it Windows gets the Linux `pypi_packages`.
        win_download_mapping: Windows override for `download_mapping`.
    """
    download_info = _unzip_archives(archives)
    win_download_info = _unzip_archives(win_archives)
    return repository_rule(
        implementation = _prebuilt_libs_repo_impl,
        environ = [
            root_env_var,
            # Arch detection for _check_download_arch and `%{arch_dir}`.
            "CC",
            "CXX",
            "PROCESSOR_ARCHITECTURE",
            "PROCESSOR_ARCHITEW6432",
        ],
        local = True,
        configure = True,
        attrs = {
            "root_env_var": attr.string(default=root_env_var),
            "urls": attr.string_list(default=download_info.urls),
            "sha256s": attr.string_list(default=download_info.sha256s),
            "strip_prefixes": attr.string_list(default=download_info.strip_prefixes),
            "download_arch": attr.string(default=download_arch),
            "win_download_arch": attr.string(default=win_download_arch or download_arch),
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
            "win_optional_libs": attr.string_list(default=win_optional_libs),
            "win_build_template": attr.label(allow_files=True,
                                             default=Label(win_build_template or build_template)),
            "win_urls": attr.string_list(default=win_download_info.urls),
            "win_sha256s": attr.string_list(default=win_download_info.sha256s),
            "win_strip_prefixes": attr.string_list(default=win_download_info.strip_prefixes),
            "pypi_packages": attr.string_list(default=pypi_packages),
            "win_pypi_packages": attr.string_list(default=win_pypi_packages),
            "_pypi_lock": attr.label(default=Label("@onedal//dev/bazel/deps:requirements_lock.txt")),
            "_download_mapping": attr.string_dict(default=download_mapping),
            "_win_download_mapping": attr.string_dict(default=win_download_mapping),
            "_fallback_mapping": attr.string_dict(default=fallback_mapping),
        }
    )

repos = struct(
    archive = _archive,
    download_and_extract = _download_and_extract,
    prebuilt_libs_repo_rule = _prebuilt_libs_repo_rule,
    create_symlinks = _create_symlinks,
)
