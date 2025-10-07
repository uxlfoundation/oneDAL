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
                repo_ctx.execute(["bash", "-c", "unzstd '%s' --stdout | tar -xf - -C '%s'" % (entry, output)])

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



def _create_download_info(repo_ctx):
    if repo_ctx.attr.url and repo_ctx.attr.urls:
        fail("Either `url` or `urls` attribute must be set")
    if repo_ctx.attr.sha256 and repo_ctx.attr.sha256s:
        fail("Either `sha256` or `sha256s` attribute must be set")
    if repo_ctx.attr.strip_prefix and repo_ctx.attr.strip_prefixes:
        fail("Either `strip_prefix` or `strip_prefixes` attribute must be set")
    if repo_ctx.attr.url:
        return struct(
            urls = [repo_ctx.attr.url],
            sha256s = [repo_ctx.attr.sha256],
            strip_prefixes = [repo_ctx.attr.strip_prefix],
        )
    else:
        return struct(
            urls = repo_ctx.attr.urls,
            sha256s = repo_ctx.attr.sha256s,
            strip_prefixes = (
                repo_ctx.attr.strip_prefixes if repo_ctx.attr.strip_prefixes else
                len(repo_ctx.attr.urls) * [repo_ctx.attr.strip_prefix]
            ),
        )

def _normalize_download_info(repo_ctx):
    info = _create_download_info(repo_ctx)
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

def _create_symlinks(repo_ctx, root, entries, substitutions={}, mapping={}):
    for entry in entries:
        entry_fmt = utils.substitute(entry, substitutions)
        src_entry_path = utils.substitute(paths.join(root, entry_fmt), mapping)
        dst_entry_path = entry_fmt
        repo_ctx.symlink(src_entry_path, dst_entry_path)

def _download(repo_ctx):
    output = repo_ctx.path("archive")
    info_entries = _normalize_download_info(repo_ctx)
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
        if repo_ctx.attr.url or repo_ctx.attr.urls:
            root = _download(repo_ctx)
            mapping = repo_ctx.attr._download_mapping
        elif repo_ctx.attr.fallback_root:
            root = repo_ctx.attr.fallback_root
        else:
            fail("Cannot locate {} dependency".format(repo_ctx.name))
    substitutions = {
        # TODO: Detect OS
        "%{os}": "lnx",
    }
    _create_symlinks(repo_ctx, root, repo_ctx.attr.includes, substitutions, mapping)
    _create_symlinks(repo_ctx, root, repo_ctx.attr.libs, substitutions, mapping)
    _create_symlinks(repo_ctx, root, repo_ctx.attr.bins, substitutions, mapping)
    repo_ctx.template(
        "BUILD",
        repo_ctx.attr.build_template,
        substitutions = substitutions,
    )

def _prebuilt_libs_repo_rule(includes, libs, build_template, bins=[],
                             root_env_var="", fallback_root="",
                             url="", sha256="", strip_prefix="",
                             local_mapping={}, download_mapping={}):
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
            "build_template": attr.label(allow_files=True,
                                         default=Label(build_template)),
            "_local_mapping": attr.string_dict(default=local_mapping),
            "_download_mapping": attr.string_dict(default=download_mapping),
        }
    )

repos = struct(
    prebuilt_libs_repo_rule = _prebuilt_libs_repo_rule,
    create_symlinks = _create_symlinks,
)
