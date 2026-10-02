#===============================================================================
# Copyright Contributors to the oneDAL Project
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

"""Make-build tasks: the agent builds with the make build system; Bazel must not be used for the build."""
import os
import re
import subprocess
import sys

from common import sh
from graders.build import g_artifact_path

BAZEL_BUILD = re.compile(r"\bbazel(?:isk)?\s+(?:-\S+\s+)*(build|test|run)\b")


def g_make_artifact(rd, ws, t, tr):
    """g_artifact_path, plus the trace must show no `bazel build/test/run` (the task forbids Bazel)."""
    g = g_artifact_path(rd, ws, t, tr)
    used_bazel = any(BAZEL_BUILD.search(c) for c in tr.get("bazel_cmds", []))
    return {**g, "used_bazel": used_bazel, "pass": bool(g["pass"] and not used_bazel)}


# Measured recipe (build_make_gnu): oneMKL static libs + headers and oneTBB from PyPI into a venv
# outside the repo, then the CI target `daal` with GCC for one ISA (sse2 is always added).
MAKE_CMD = ["make", "-f", "makefile", "daal", "PLAT=lnx32e", "COMPILER=gnu", "REQCPU=avx2"]


def o_make(rd, repo, t):
    deps = rd / "deps"
    sh([sys.executable, "-m", "venv", str(deps)])
    sh([str(deps / "bin" / "pip"), "install", "-q", "mkl-static", "mkl-include", "tbb-devel"], timeout=1800)
    env = dict(os.environ, MKLROOT=str(deps), TBBROOT=str(deps),
               LD_LIBRARY_PATH=f"{deps}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}")
    cmd = [*MAKE_CMD, f"-j{os.cpu_count()}"]
    with open(rd / "oracle_make.log", "w") as log:
        subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    lib = next(iter(sorted(repo.glob("__release_lnx_gnu/daal/latest/lib/intel64/libonedal_core.so*"))), None)
    return '```json\n{"path": "' + str(lib) + '", "command": "' + " ".join(cmd) + '"}\n```'


GRADERS = {"make_artifact": g_make_artifact}
ORACLES = {"make_artifact": o_make}
