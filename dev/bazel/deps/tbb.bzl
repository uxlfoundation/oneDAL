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

tbb_repo = repos.prebuilt_libs_repo_rule(
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
