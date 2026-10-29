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

_VERSION = "2021.16.2"

ccl_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "CCL_ROOT",
    archives = [
        repos.archive(
            url = "https://files.pythonhosted.org/packages/20/84/80906846a15a688214479b30c5a25044f3356216f6ee473b5f33010d7db0/oneccl-{}-py2.py3-none-manylinux_2_28_x86_64.whl".format(_VERSION),
            sha256 = "83fa5c0ac8e7c06569480fd27eb5f26a3bb383902277e2d22c93212b6cf26e9e",
            strip_prefix = "oneccl-{}.data/data".format(_VERSION),
        ),
    ],
    includes = [
        "include/cpu_gpu_dpcpp/oneapi/",
    ],
    libs = [
        "lib/cpu_gpu_dpcpp/libccl.a",
        "lib/cpu_gpu_dpcpp/libccl.so",
        "lib/cpu_gpu_dpcpp/libccl.so.1",
        "lib/cpu_gpu_dpcpp/libccl.so.1.0",
    ],
    build_template = "@onedal//dev/bazel/deps:ccl.tpl.BUILD",

)

def _get_fi_providers_dir(fi_files):
    if len(fi_files) == 0:
        fail("No fabrin interface files provided for MPI")
    # Runfiles paths: external repositories are siblings of the main one there.
    fi_dir = fi_files[0].short_path.rpartition("/")[0]
    for fi in fi_files:
        if fi.short_path.rpartition("/")[0] != fi_dir:
            fail("All fabric interface files must reside in the same directory")
    return fi_dir

def _generate_mpiexec_wrapper(ctx, mpiexec, executable, fi_dir, libfabric):
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
        "export FI_PROVIDER_PATH=\"$(pwd)/{}\"\n".format(fi_dir) +
        # The ranks run on one node. Without this, Intel MPI may pick a NIC
        # provider (seen: verbs/psm3 on CI VMs) that cannot open an endpoint.
        "export I_MPI_FABRICS=\"${I_MPI_FABRICS:-shm}\"\n" +
        # libmpi dlopens libfabric.so.1 and has no RUNPATH of its own.
        "export LD_LIBRARY_PATH=\"$(pwd)/{}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\"\n".format(
            libfabric.short_path.rpartition("/")[0]) +
        # Catch2 seeds each rank on its own; the ranks must generate the same data.
        "{} -n {} {} --rng-seed \"${{RANDOM}}\" \"$@\"\n".format(mpiexec.short_path,
                                      ctx.attr.mpi_ranks,
                                      executable.short_path)
    )
    ctx.actions.write(exec_wrapper, content, is_executable=True)
    return exec_wrapper


def _ccl_test_impl(ctx):
    exec = ctx.executable.src
    mpiexec = ctx.files.mpiexec[0]
    fi_files = ctx.files.fi
    fi_dir = _get_fi_providers_dir(fi_files)
    exec_wrapper = _generate_mpiexec_wrapper(ctx, mpiexec, exec, fi_dir, ctx.file.libfabric)
    return DefaultInfo(
        files = depset([ exec_wrapper ]),
        runfiles = ctx.runfiles(
            files = fi_files + [ exec, ctx.file.libfabric ],
            transitive_files = ctx.attr.mpiexec.default_runfiles.files,
        ),
        executable = exec_wrapper,
    )

ccl_test = rule(
    implementation = _ccl_test_impl,
    attrs = {
        # The test binary itself: built for the target like any other test,
        # so it shares the build with the rest of the test suite.
        "src": attr.label(mandatory=True,
                          executable=True,
                          cfg="target"),
        "mpi_ranks": attr.int(mandatory=True),
        "mpiexec": attr.label(mandatory=True,
                              executable=True,
                              cfg="exec"),
        "fi": attr.label(mandatory=True),
        "libfabric": attr.label(mandatory=True, allow_single_file=True),
    },
    test = True,
)
