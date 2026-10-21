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

"""collect_bazel.py <unit>  -> data/cov2/<unit>.json.gz

Same output as collect.py, for oneDAL's own Bazel tests (bazel_tests.sh) built
with gcc --coverage on the cov2 commit. The .gcno files sit in the Bazel
execroot; the tests wrote .gcda under GCOV_PREFIX + /proc/self/cwd/<obj>.
Source paths are execroot-relative (cpp/...), so canon() maps them onto the
same files as the Make build; matching to the universe is by mangled name and
file:line.
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

UNIT = sys.argv[1]
sys.argv = [sys.argv[0], "cov2"]
import collect  # noqa: E402

EXEC = glob.glob(f"{collect.U}/bzroot/*/execroot/_main")[0]
GCDA = f"{collect.U}/cov-cov2/u/{UNIT}/proc/self/cwd/"


def main():
    os.makedirs(f"{collect.U}/gcov-work", exist_ok=True)
    gcnos = []
    for d, _, fs in os.walk(f"{EXEC}/bazel-out/k8-opt/bin/cpp"):
        gcnos += [os.path.join(d, f) for f in fs if f.endswith(".gcno")]
    collect.gcov_one = lambda gcno, _root: _gcov(gcno)
    collect.collect(UNIT, gcnos)


def _gcov(gcno):
    gcda = GCDA + os.path.relpath(gcno, EXEC)[:-5] + ".gcda"
    if not os.path.exists(gcda):
        return None
    with tempfile.TemporaryDirectory(dir=f"{collect.U}/gcov-work") as td:
        base = os.path.join(td, "o")
        os.symlink(gcno, base + ".gcno")
        os.symlink(gcda, base + ".gcda")
        r = subprocess.run(collect.GCOV + ["--json-format", "--stdout", "--demangled-names", "-o", td, base],
                           cwd=EXEC, capture_output=True, text=True)
    return [json.loads(ln) for ln in r.stdout.splitlines() if ln.strip().startswith("{")]


if __name__ == "__main__":
    main()
