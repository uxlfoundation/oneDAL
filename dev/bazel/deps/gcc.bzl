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

# The conda-forge GNU toolchain used on an x86-64 Linux host that has no C++
# compiler. It brings its own binutils and a glibc 2.28 sysroot, so neither the
# host compiler nor the host libc headers are used.

load("@onedal//dev/bazel:repos.bzl", "repos")

# Bumped by Renovate (conda datasource). Builds and hashes of the packages below
# are re-pinned by `python3 dev/bazel/deps/relock_conda.py`.
_GCC_VERSION = "14.3.0"
_TRIPLE = "x86_64-conda-linux-gnu"
_URL = "https://anaconda.org/conda-forge/{name}/{version}/download/{subdir}/{name}-{version}-{build}.conda"

def _pkg(name, version, build, sha256, subdir = "linux-64"):
    return struct(
        url = _URL.format(name = name, version = version, build = build, subdir = subdir),
        sha256 = sha256,
    )

_PACKAGES = [
    _pkg("gcc_impl_linux-64", _GCC_VERSION, "h054831b_20", "09bb9b0d54b012c36a115e3ecebc80276df261555e847afbb3fbc7ab5566408c"),
    _pkg("gxx_impl_linux-64", _GCC_VERSION, "h99ea42b_20", "8cd46d6f94a5a34181492549e1e56e5795337007289e6ca30d9ff1864f0e33c1"),
    _pkg("libgcc-devel_linux-64", _GCC_VERSION, "hf649bbc_120", "9e2a3e7de26fc149707f9ff3988dc3e6a0b9534aea1bedcaa6ec9ba93dfdd013", subdir = "noarch"),
    _pkg("libstdcxx-devel_linux-64", _GCC_VERSION, "h9f08a49_120", "5b7fd889a648c71d71a7cdda09107c0f189aa306f77c33cd66678a56cb225e53", subdir = "noarch"),
    _pkg("libgcc", _GCC_VERSION, "he0feb66_20", "ee549953022da53692ae9a5da17a5d4a1829b9a173b68433e529279f0d2f9305"),
    _pkg("libstdcxx", _GCC_VERSION, "h934c35e_20", "83447d514a21052674806585721fef05979dd6700ca199dbfdef713b6113cb2b"),
    _pkg("libgomp", _GCC_VERSION, "he0feb66_20", "b182215de444505bf9d590d409cc685b2fc1510ea5e865ffb2e0c6d4619cd01c"),
    _pkg("libsanitizer", _GCC_VERSION, "h91d2232_20", "4b685b1da0f85f4771e1b243e66cf2cf7b2e625caa440efc082d800a4e9b091e"),
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
        archive = repo_ctx.path(pkg.url.split("/")[-1])
        repo_ctx.download(url = pkg.url, output = archive, sha256 = pkg.sha256)
        repos.extract_conda(repo_ctx, archive, prefix)
        repo_ctx.delete(archive)
    _drop_build_rpath(repo_ctx, prefix)
    tool = lambda name: str(repo_ctx.path("{}/bin/{}-{}".format(prefix, _TRIPLE, name)))
    return struct(cc = tool("gcc"), ar = tool("ar"), strip = tool("strip"))
