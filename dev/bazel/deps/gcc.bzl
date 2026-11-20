#===============================================================================
# Copyright contributors to the oneDAL project
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

# The conda-forge GNU toolchain for an x86-64 Linux host without a C++ compiler. It brings its
# own binutils and a glibc 2.28 sysroot, so nothing of the host toolchain is used.

load("@onedal//dev/bazel:repos.bzl", "repos")

# Bumped by Renovate (conda datasource); the builds and hashes below are
# re-pinned by dev/bazel/deps/relock_conda.py (see dev/bazel/README.md).
_GCC_VERSION = "14.4.0"
_TRIPLE = "x86_64-conda-linux-gnu"
_URL = "https://anaconda.org/conda-forge/{name}/{version}/download/{subdir}/{name}-{version}-{build}.conda"

def _pkg(name, version, build, sha256, subdir = "linux-64"):
    url = _URL.format(name = name, version = version, build = build, subdir = subdir)
    return repos.archive(url = url, sha256 = sha256)

_PACKAGES = [
    _pkg("gcc_impl_linux-64", _GCC_VERSION, "h4b2a866_7", "fe3d0c2cbd59755ddd8364c79b8e1a09b3125581d9df07170f102bc150740632"),
    _pkg("gxx_impl_linux-64", _GCC_VERSION, "hc436dd5_7", "8082bf31bf210f55ab3e69bbc301e5131b8fe2986c743b4d4486d469dc737ac3"),
    _pkg("libgcc-devel_linux-64", _GCC_VERSION, "hd9a9cd0_107", "fb09fdc5d12cc929ff0af73fbd7cf33be7b473a46fe11353a21ec1ffa468c657", subdir = "noarch"),
    _pkg("libstdcxx-devel_linux-64", _GCC_VERSION, "ha5b54cb_107", "a518b2878cecd324082d334d36907156b68074b18736c111b15ff34efcb22aaf", subdir = "noarch"),
    _pkg("libgcc", _GCC_VERSION, "hecc6394_7", "b8ae5d04510e014db4e4e04ada048479f89e507fac96e9b7572fc3520e943db6"),
    _pkg("libstdcxx", _GCC_VERSION, "h934c35e_7", "c8dce64507240af2f6132518400448080697a99c317989aff3e1ae8fc01cbdf1"),
    _pkg("libgomp", _GCC_VERSION, "he0feb66_7", "76b34fc094732e38055df85f99acdfc668eacbe38aaa81b100ddea454c0e29df"),
    _pkg("libsanitizer", _GCC_VERSION, "hf39dbba_7", "7a4c47d699e6e9640aa8061f461ab31e71ba7d41994a451ea1e79a3801e8f6c3"),
    _pkg("binutils_impl_linux-64", "2.46.1", "default_hfdba357_103", "98f9ddc6dc5d56a824b7eaea5bab6ad4bb47642288fc70d6b1b4024ac39d5595"),
    _pkg("ld_impl_linux-64", "2.46.1", "default_hbd61a6d_103", "901649b1edbe841fd4ee7e2d491a05184ed540eb01cdae506872f344babc1d28"),
    _pkg("zstd", "1.5.7", "hb78ec9c_7", "47d682b9f6d6ec9eb1a6e6c3e75ea6273e899e78fb7fc59f81d39745009fbc60"),
    _pkg("sysroot_linux-64", "2.28", "he3f20f0_9", "e2c4c08188bc543756ce2cf596b9547a241398b0ddce1cef77aa93847f22a97e", subdir = "noarch"),
    _pkg("kernel-headers_linux-64", "4.18.0", "he073ed8_9", "41557eeadf641de6aeae49486cef30d02a6912d8da98585d687894afd65b356a", subdir = "noarch"),
]

# GNU OpenMP for `@openmp`, which takes it from this pin so that one Renovate
# bump (and one relock_conda.py run) moves the compiler and its runtime together.
LIBGOMP = [pkg for pkg in _PACKAGES if "/libgomp-" in pkg.url][0]

def _drop_build_rpath(repo_ctx, prefix):
    # conda rewrites its build prefix at install time; here it would leak into
    # every linked binary as `-rpath <build prefix>/lib`.
    specs = repo_ctx.path("{}/lib/gcc/{}/{}/specs".format(prefix, _TRIPLE, _GCC_VERSION))
    content = repo_ctx.read(specs)
    start = content.find("%{!static:-rpath ")
    if start < 0:
        fail("No build-prefix rpath in {}; the conda gcc package layout has changed".format(specs))
    end = content.find("}", start)
    repo_ctx.file(specs, content[:start] + content[end + 1:], executable = False)

def download_gcc(repo_ctx, prefix):
    """Downloads the GNU toolchain into `prefix` of the repository.

    Args:
        repo_ctx: Repository context of the toolchain rule.
        prefix: Directory, relative to the repository root, to unpack into.

    Returns:
        A struct with the absolute paths of `cc`, `ar` and `strip`.
    """
    for pkg in _PACKAGES:
        repos.download_and_extract(repo_ctx, pkg.url, pkg.sha256, prefix, pkg.strip_prefix)
    _drop_build_rpath(repo_ctx, prefix)
    tool = lambda name: str(repo_ctx.path("{}/bin/{}-{}".format(prefix, _TRIPLE, name)))
    return struct(cc = tool("gcc"), ar = tool("ar"), strip = tool("strip"))
