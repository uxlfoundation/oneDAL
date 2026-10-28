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

_VERSION = "2021.16.1"

mpi_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "MPIROOT",
    archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/e5/44/6867aebd60d8b8bcf8f7e3b1fa781debd4fc3df16bfb1525e732570ea214/impi_rt-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "0f70499bc42eab6923c0ae8fa6dd409c047d93626d686ac21d04e295b120bb95",
            strip_prefix = "impi_rt-{}.data/data".format(_VERSION),
        ),
        repos.archive(
            url = "https://files.pythonhosted.org/packages/41/75/46b81412f336dde770a56960c7b10af6860efa4dd3aa9796d5eb2645644a/impi_devel-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "d1081a1647a4de521e2680983629d627704364a07f16c60aaec0bb85535e25bd",
            strip_prefix = "impi_devel-{}.data/data".format(_VERSION),
        ),
    ],
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
    build_template = "@onedal//dev/bazel/deps:mpi.tpl.BUILD",
    download_mapping = {
        # Required directory layout and layout in the downloaded
        # archives may be different. Mapping helps to setup relations
        # between the required layout (LHS) and downloaded (RHS).
        #          REQUIRED                              DOWNLOADED
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

def _get_fi_providers_dir(fi_files):
    if len(fi_files) == 0:
        fail("No fabrin interface files provided for MPI")
    fi_dir = fi_files[0].dirname
    for fi in fi_files:
        if fi.dirname != fi_dir:
            fail("All fabric interface files must reside in the same directory")
    return fi_dir

def _generate_mpiexec_wrapper(ctx, mpiexec, executable, fi_dir):
    exec_wrapper = ctx.actions.declare_file(ctx.label.name)
    content = (
        "#!/bin/bash\n" +
        "# We need to check if we are in the runfiles directory.\n" +
        "# If no change current directory to runfiles.\n" +
        "runfiles_suffix=\".runfiles/{}\"\n".format(ctx.workspace_name) +
        "if [[ ! \"$(pwd)\" =~ \"$runfiles_suffix\" ]]; then\n" +
        "   script_path=\"${BASH_SOURCE[0]}\"\n" +
        "   cd ${script_path}${runfiles_suffix}\n" +
        "fi\n" +
        "export FI_PROVIDER_PATH=\"{}\"\n".format(fi_dir) +
        "{} -n {} {} \"$@\"\n".format(mpiexec.path,
                                      ctx.attr.mpi_ranks,
                                      executable.short_path)
    )
    ctx.actions.write(exec_wrapper, content, is_executable=True)
    return exec_wrapper

def _mpi_test_impl(ctx):
    exec = ctx.executable.src
    mpiexec = ctx.files.mpiexec[0]
    fi_files = ctx.files.fi
    fi_dir = _get_fi_providers_dir(fi_files)
    exec_wrapper = _generate_mpiexec_wrapper(ctx, mpiexec, exec, fi_dir)
    return DefaultInfo(
        files = depset([ exec_wrapper ]),
        runfiles = ctx.runfiles(
            files = fi_files + [ exec ],
            transitive_files = ctx.attr.mpiexec.default_runfiles.files,
        ),
        executable = exec_wrapper,
    )

mpi_test = rule(
    implementation = _mpi_test_impl,
    attrs = {
        "src": attr.label(mandatory=True,
                          executable=True,
                          cfg="exec"),
        "mpi_ranks": attr.int(mandatory=True),
        "mpiexec": attr.label(mandatory=True,
                              executable=True,
                              cfg="exec"),
        "fi": attr.label(mandatory=True),
    },
    test = True,
)
