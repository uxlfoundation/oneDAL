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

"""collect_gpu.py <unit|universe|all>  -> data/gpu/<name>.json.gz

collect.py for the icpx build in oneDAL-gpu: clang writes gcov notes that gcc's
gcov cannot read, and llvm-cov gcov has no JSON output, so this parses its
intermediate text format (-i) into the same document shape. Function keys are
demangled names (the intermediate format carries no mangled ones). Host code
only: SYCL device code is not instrumented.
"""
import glob
import os
import subprocess
import sys
import tempfile

ARGS = sys.argv[2:] if len(sys.argv) > 2 else sys.argv[1:]
sys.argv = [sys.argv[0], "gpu"]
import collect  # noqa: E402

LLVM_COV = f"{collect.U}/cenv/bin/compiler/llvm-cov"


def parse(text_files):
    files = {}
    for path in text_files:
        cur = None
        for line in open(path, errors="replace"):
            tag, _, rest = line.rstrip("\n").partition(":")
            if tag == "file":
                cur = files.setdefault(rest, dict(file=rest, functions=[], lines=[]))
            elif tag == "function" and cur is not None:
                start, count, name = rest.split(",", 2)
                cur["functions"].append(dict(name=name, demangled_name=name, start_line=int(start),
                                             execution_count=int(count)))
            elif tag == "lcount" and cur is not None:
                ln, count = rest.split(",")[:2]
                cur["lines"].append(dict(line_number=int(ln), count=int(count)))
    return [dict(files=list(files.values()))]


def gcov_one(gcno, gcda_root):
    obj = gcno[:-5]
    with tempfile.TemporaryDirectory(dir=f"{collect.U}/gcov-work") as td:
        base = os.path.join(td, "o")
        os.symlink(gcno, base + ".gcno")
        if gcda_root:
            gcda = f"{gcda_root}{obj}.gcda"
            if not os.path.exists(gcda):
                return None
            os.symlink(gcda, base + ".gcda")
        # -i writes one .gcov per source into the cwd, so run inside the temp dir
        subprocess.run([LLVM_COV, "gcov", "-i", "-m", "-o", td, base + ".gcno"],
                       cwd=td, capture_output=True, text=True)
        return parse(glob.glob(f"{td}/*.gcov"))


def main():
    collect.gcov_one = gcov_one
    os.makedirs(f"{collect.U}/gcov-work", exist_ok=True)
    gcnos = []
    for w in collect.WORKS:
        for d, _, fs in os.walk(w):
            gcnos += [os.path.join(d, f) for f in fs if f.endswith(".gcno") and "_static" not in d]
    names = ARGS
    if names == ["all"]:
        names = ["universe"] + sorted(os.listdir(f"{collect.U}/cov-gpu/u"))
    for n in names:
        collect.collect(n, gcnos)


if __name__ == "__main__":
    main()
