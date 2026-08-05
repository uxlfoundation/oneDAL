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

"""Bazel `rule()` that wraps a test binary in an mpiexec launcher.

Lives here (dev/bazel/rules/) rather than in the @mpi repository because
BUILD-time custom rules must be loadable from the main workspace; the @mpi
external repository is only responsible for exposing prebuilt mpi headers
and shared libraries.
"""

def _get_fi_providers_dir(fi_files):
    if len(fi_files) == 0:
        fail("No fabric interface files provided for MPI")
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
        "# If not, change current directory to runfiles.\n" +
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
