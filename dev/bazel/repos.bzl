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

def _create_download_info(repo_ctx, os_id):
    url = _select_by_os(repo_ctx, "url", os_id)
    urls = _select_by_os(repo_ctx, "urls", os_id)
    sha256 = _select_by_os(repo_ctx, "sha256", os_id)
    sha256s = _select_by_os(repo_ctx, "sha256s", os_id)
    strip_prefix = _select_by_os(repo_ctx, "strip_prefix", os_id)
    strip_prefixes = _select_by_os(repo_ctx, "strip_prefixes", os_id)
    if url and urls:
        fail("Either `url` or `urls` attribute must be set")
    if sha256 and sha256s:
        fail("Either `sha256` or `sha256s` attribute must be set")
    if strip_prefix and strip_prefixes:
        fail("Either `strip_prefix` or `strip_prefixes` attribute must be set")
    if url:
        return struct(
            urls = [url],
            sha256s = [sha256],
            strip_prefixes = [strip_prefix],
        )
    else:
        return struct(
            urls = urls,
            sha256s = sha256s,
            strip_prefixes = (
                strip_prefixes if strip_prefixes else
                len(urls) * [strip_prefix]
            ),
        )

def _normalize_download_info(repo_ctx, os_id):
    info = _create_download_info(repo_ctx, os_id)
    expected_len = len(info.urls)
    if len(info.sha256s) != expected_len:
        fail("sha256 hashes count does not match URLs count")
    if len(info.strip_prefixes) != expected_len:
        fail("strip_prefixes count does not match URLs count")
    result = []
    for url, sha256, strip_prefix in zip(info.urls, info.sha256s, info.strip_prefixes):
        result.append(struct(
            url = url,
            sha256 = sha256,
            strip_prefix = strip_prefix
        ))
    return result

def _create_symlinks(repo_ctx, root, entries, substitutions=None, mapping=None):
    substitutions = substitutions or {}
    mapping = mapping or {}

    for entry in entries:
        entry_fmt = utils.substitute(entry, substitutions)
        if "*" in entry_fmt:
            pattern = entry_fmt.split("/")[-1]
            dir_part = entry_fmt[:entry_fmt.rfind("/")] if "/" in entry_fmt else ""
            root_with_dir = utils.substitute(
                paths.join(root, dir_part) if dir_part else root,
                mapping
            )
            matched = False
            for fs_entry in repo_ctx.path(root_with_dir).readdir():
                if _matches_glob(fs_entry.basename, pattern):
                    matched = True
                    dst = (paths.join(dir_part, fs_entry.basename)
                           if dir_part else fs_entry.basename)
                    repo_ctx.symlink(str(fs_entry), dst)
            if not matched:
                fail("No files matched pattern '%s' in directory '%s' while creating symlinks for entry '%s'" %
                     (pattern, root_with_dir, entry_fmt))
        else:
            src_entry_path = utils.substitute(paths.join(root, entry_fmt), mapping)
            dst_entry_path = entry_fmt
            repo_ctx.symlink(src_entry_path, dst_entry_path)

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

# TODO: Delete hardcoded package keywords after release
def _prebuilt_libs_repo_impl(repo_ctx):
    os_id = _detect_os(repo_ctx)
    root = repo_ctx.os.environ.get(repo_ctx.attr.root_env_var)
    if root:
        if "2017u1" in root:
            mapping = repo_ctx.attr._local_mapping
        elif "2023u1" in root:
            mapping = repo_ctx.attr._local_mapping
        elif "20230413" in root:
            mapping = repo_ctx.attr._local_mapping
        elif "2021.10.0-RC" in root:
            mapping = repo_ctx.attr._local_mapping
        elif "2021.2-gold_236" in root:
            mapping = repo_ctx.attr._local_mapping
        else:
            mapping = {}
    else:
        if _select_by_os(repo_ctx, "url", os_id) or _select_by_os(repo_ctx, "urls", os_id):
            root = _download(repo_ctx, os_id)
            mapping = _select_by_os(repo_ctx, "_download_mapping", os_id)
        elif repo_ctx.attr.fallback_root:
            root = repo_ctx.attr.fallback_root
        else:
            fail("Cannot locate {} dependency".format(repo_ctx.name))
    substitutions = {
        "%{os}": os_id,
    }
    # Extract substitutions if a makefile_ver attribute is present
    if hasattr(repo_ctx.attr, "_makefile_ver") and repo_ctx.attr._makefile_ver:
        makefile_ver = repo_ctx.path(repo_ctx.attr._makefile_ver)
        makefile_content = repo_ctx.read(makefile_ver)
        binary_major = "4"
        binary_minor = "0"
        for line in makefile_content.splitlines():
            if line.startswith("MAJORBINARY"):
                binary_major = line.split("=")[1].strip()
            elif line.startswith("MINORBINARY"):
                binary_minor = line.split("=")[1].strip()
        substitutions["%{version_binary_major}"] = binary_major
        substitutions["%{version_binary_minor}"] = binary_minor

    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "includes", os_id), substitutions, mapping)
    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "libs", os_id), substitutions, mapping)
    _create_symlinks(repo_ctx, root, _select_by_os(repo_ctx, "bins", os_id), substitutions, mapping)
    repo_ctx.template(
        "BUILD",
        _select_by_os(repo_ctx, "build_template", os_id),
        substitutions = substitutions,
    )

def _prebuilt_libs_repo_rule(includes, libs, build_template, bins=[],
                             root_env_var="", fallback_root="",
                             url="", sha256="", strip_prefix="",
                             local_mapping={}, download_mapping={},
                             win_includes=[], win_libs=[], win_bins=[], win_build_template=None,
                             win_url="", win_urls=[], win_sha256="", win_sha256s=[],
                             win_strip_prefix="", win_strip_prefixes=[], win_download_mapping={}):
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
            "url": attr.string(default=url),
            "urls": attr.string_list(default=[]),
            "sha256": attr.string(default=sha256),
            "sha256s": attr.string_list(default=[]),
            "strip_prefix": attr.string(default=strip_prefix),
            "strip_prefixes": attr.string_list(default=[]),
            "includes": attr.string_list(default=includes),
            "libs": attr.string_list(default=libs),
            "bins": attr.string_list(default=bins),
            "_makefile_ver": attr.label(default=Label("@onedal//:makefile.ver")),
            "build_template": attr.label(allow_files=True,
                                         default=Label(build_template)),
            "win_includes": attr.string_list(default=win_includes),
            "win_libs": attr.string_list(default=win_libs),
            "win_bins": attr.string_list(default=win_bins),
            "win_build_template": attr.label(allow_files=True,
                                             default=Label(win_build_template or build_template)),
            "win_url": attr.string(default=win_url),
            "win_urls": attr.string_list(default=win_urls),
            "win_sha256": attr.string(default=win_sha256),
            "win_sha256s": attr.string_list(default=win_sha256s),
            "win_strip_prefix": attr.string(default=win_strip_prefix),
            "win_strip_prefixes": attr.string_list(default=win_strip_prefixes),
            "_local_mapping": attr.string_dict(default=local_mapping),
            "_download_mapping": attr.string_dict(default=download_mapping),
            "_win_download_mapping": attr.string_dict(default=win_download_mapping),
        }
    )

repos = struct(
    prebuilt_libs_repo_rule = _prebuilt_libs_repo_rule,
    prebuilt_libs_repo_impl = _prebuilt_libs_repo_impl,
    create_symlinks = _create_symlinks,
)
