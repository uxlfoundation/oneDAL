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

load("@onedal//dev/bazel:repos.bzl", "repos")

_VERSION = "2023.0.0"

tbb_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "TBBROOT",
    archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/aa/d2/9a994ce9b18182b04783282eba77e236d23919acf42a886d72fe14fc78a4/tbb-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "482f57656386ea14b96e8da36b3fcc4cd880834ef0f328ba09e8e2e3c639285e",
            strip_prefix = "tbb-{}.data/data".format(_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/3d/ce/3f7551583ad858dc6d6b8a6b34f1843db3887879677cb772fdc6101aaa1a/tbb_devel-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "612eb3f7dd2096868ac9907c9e3fb2246f8d0fe88a7c5531163e5c4515bf801b",
            strip_prefix = "tbb_devel-{}.data/data".format(_VERSION),
        ),
    ],
    win_archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/7a/f8/dd5e1d9955c91ebd301d855232163750489b20ea5a8c4e40954269493dcc/tbb-{}-py3-none-win_amd64.whl".format(_VERSION),
            sha256 = "82d1c2b4b6881d5b4f2e67288ca1c864f54e3cd82c7e428aaf912aa30ab21d01",
            strip_prefix = "tbb-{}.data/data/Library".format(_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/db/da/07298ded5d56686fce1e0123730d6a5e288f7951863beb4fcc0214ad8ed0/tbb_devel-{}-py3-none-win_amd64.whl".format(_VERSION),
            sha256 = "acc9df3c4f60b662dcd1fa27f72e40580201f30931fb8bd7336c603c8f4f1664",
            strip_prefix = "tbb_devel-{}.data/data/Library".format(_VERSION),
        ),
    ],
    includes = [
        "include",
    ],
    libs = [
        "lib/libtbb.so",
        "lib/libtbb.so.12",
        "lib/libtbbmalloc.so",
        "lib/libtbbmalloc.so.2",
    ],
    build_template = "@onedal//dev/bazel/deps:tbb.tpl.BUILD",
    win_includes = [
        "include",
    ],
    # Globs rather than exact names so the `_debug` variants required by an
    # `-MDd` build (`tbb12_debug.lib`, see makefile:330) are picked up when the
    # TBB layout ships them, without making the default release build fail on
    # installations that do not. `tbb_win.tpl.BUILD` names the debug files
    # explicitly in its `select()`, so a debug-runtime build that is missing
    # them fails with a plain missing-input error instead of silently
    # dropping the dependency.
    win_libs = [
        "lib/tbb12*.lib",
        "lib/tbbmalloc*.lib",
    ],
    win_bins = [
        "bin/tbb12*.dll",
        "bin/tbbmalloc*.dll",
    ],
    win_build_template = "@onedal//dev/bazel/deps:tbb_win.tpl.BUILD",
)
