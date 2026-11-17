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

mpi_repo = repos.prebuilt_libs_repo_rule(
    root_env_var = "MPIROOT",
    # Pinned in requirements_lock.txt.
    pypi_packages = [
        "impi-rt",
        "impi-devel",
        # impi-rt no longer bundles libfabric; it depends on this wheel instead.
        "modified-libfabric",
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
    ],
    # The provider set changes between libfabric releases. The Windows wheels
    # ship none, and nothing on Windows runs mpiexec.
    required_any_libs = [
        "libfabric/lib/prov/*-fi.so",
    ],
    required_any_libs_description = "libfabric providers, which mpiexec loads through FI_PROVIDER_PATH",
    build_template = "@onedal//dev/bazel/deps:mpi.tpl.BUILD",
    download_mapping = {
        #          REQUIRED                              DOWNLOADED
        "libfabric/lib/libfabric.so":                   "lib/libfabric.so",
        "libfabric/lib/libfabric.so.1":                 "lib/libfabric.so.1",
        "lib/release/libmpi.so":                        "lib/libmpi.so",
        "lib/release/libmpi.so.12":                     "lib/libmpi.so.12",
        "lib/release/libmpi.so.12.0":                   "lib/libmpi.so.12.0",
        "lib/release/libmpi.so.12.0.0":                 "lib/libmpi.so.12.0.0",
        "libfabric/lib/prov":                           "lib/libfabric",
    },
)

def _get_fi_providers_dir(fi_files):
    if len(fi_files) == 0:
        fail("No fabric interface files provided for MPI")
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
        # `bazel test` and `bazel run` start in the runfiles; a direct run from bazel-bin does not.
        "runfiles_suffix=\".runfiles/{}\"\n".format(ctx.workspace_name) +
        "if [[ ! \"$(pwd)\" =~ \"$runfiles_suffix\" ]]; then\n" +
        "   cd \"${BASH_SOURCE[0]}${runfiles_suffix}\"\n" +
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

def _mpi_test_impl(ctx):
    exec = ctx.executable.src
    # The sh_binary's file is `bin/mpiexec` itself; hydra looks for hydra_bstrap_proxy beside argv[0],
    # so the launcher symlink in bazel-bin must not be what runs.
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

# Runs `src` under `mpiexec` with `mpi_ranks` ranks; the CCL tests use it too (ccl.bzl).
mpi_test = rule(
    implementation = _mpi_test_impl,
    attrs = {
        # Built for the target, so it shares the build with the rest of the test suite.
        "src": attr.label(mandatory=True,
                          executable=True,
                          cfg="target"),
        "mpi_ranks": attr.int(mandatory=True),
        # Runs at test time on the target machine, not during the build.
        "mpiexec": attr.label(mandatory=True,
                              executable=True,
                              cfg="target"),
        "fi": attr.label(mandatory=True),
        "libfabric": attr.label(mandatory=True, allow_single_file=True),
    },
    test = True,
)
