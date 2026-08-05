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

"""External repo rule for Intel MPI.

Repo-side symbol only. The mpi_test rule (a BUILD-time Bazel `rule()`) lives
in //dev/bazel/rules:mpi_test.bzl because it must be loadable from the main
workspace, not from an external repository.
"""

load("@onedal//third_party:repo.bzl", "repos")

mpi_repo = repos.prebuilt_libs_repo_rule(
    bins = [
        "bin",
    ],
    includes = [
        "include",
    ],
    libs = [
        "lib/release/libmpi.so",
        "lib/release/libmpi.so.12",
        "lib/release/libmpi.so.12.0",
        "lib/release/libmpi.so.12.0.0",
        "libfabric/lib/libfabric.so",
        "libfabric/lib/libfabric.so.1",
        "libfabric/lib/prov/libefa-fi.so",
        "libfabric/lib/prov/libmlx-fi.so",
        "libfabric/lib/prov/libpsm3-fi.so",
        "libfabric/lib/prov/libpsmx2-fi.so",
        "libfabric/lib/prov/librxm-fi.so",
        "libfabric/lib/prov/libshm-fi.so",
        "libfabric/lib/prov/libtcp-fi.so",
        "libfabric/lib/prov/libverbs-1.1-fi.so",
        "libfabric/lib/prov/libverbs-1.12-fi.so",
    ],
    build_template = "@onedal//third_party/mpi:mpi.tpl.BUILD",
    download_mapping = {
        # Required layout and downloaded-archive layout may differ.
        # LHS = required (from bazel's perspective), RHS = actual downloaded path.
        "libfabric/lib/libfabric.so":                   "lib/libfabric.so",
        "libfabric/lib/libfabric.so.1":                 "lib/libfabric.so.1",
        "lib/release/libmpi.so":                        "lib/libmpi.so",
        "lib/release/libmpi.so.12":                     "lib/libmpi.so.12",
        "lib/release/libmpi.so.12.0":                   "lib/libmpi.so.12.0",
        "lib/release/libmpi.so.12.0.0":                 "lib/libmpi.so.12.0.0",
        "libfabric/lib/prov/libefa-fi.so":              "lib/libefa-fi.so",
        "libfabric/lib/prov/libmlx-fi.so":              "lib/libmlx-fi.so",
        "libfabric/lib/prov/libpsm3-fi.so":             "lib/libpsm3-fi.so",
        "libfabric/lib/prov/libpsmx2-fi.so":            "lib/libpsmx2-fi.so",
        "libfabric/lib/prov/librxm-fi.so":              "lib/librxm-fi.so",
        "libfabric/lib/prov/libshm-fi.so":              "lib/libshm-fi.so",
        "libfabric/lib/prov/libtcp-fi.so":              "lib/libtcp-fi.so",
        "libfabric/lib/prov/libverbs-1.1-fi.so":        "lib/libverbs-1.1-fi.so",
        "libfabric/lib/prov/libverbs-1.12-fi.so":       "lib/libverbs-1.12-fi.so",
    },
)
