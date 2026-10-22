#!/bin/bash
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
# GPU SPMD tests, one unit per test file under 4 ranks. Each file is capped at CAP s
# with SIGINT so pytest exits cleanly and gcov still writes the .gcda files.
D=$(cd "$(dirname "$0")" && pwd)
U=${USAGE_MAP_WORK:-$PWD}
S=$U/sklearnex-gpusrc
CAP=${CAP:-600}
export OD=oneDAL-gpu I_MPI_FABRICS=shm PATH=$U/genv/bin:$PATH
one() {
  f=$1
  n=gpu_spmd_$(basename "$f" .py | sed 's/^test_//; s/_spmd$//')
  timeout -s INT -k 60 "$CAP" "$U/genv/bin/mpirun" -n 4 "$D/run_gpu.sh" "u/$n" "$S/tests/helper_mpi_tests.py" \
    pytest -q --with-mpi --timeout=300 -c "$S/setup.cfg" "$f" > "$U/spmd-logs/$n.log" 2>&1
  echo "$n rc=$? $(grep -E '[0-9]+ (passed|failed)' "$U/spmd-logs/$n.log" | tail -1)"
}
export -f one
export U S D CAP
mkdir -p "$U/spmd-logs"
find "$S/sklearnex/spmd" -name 'test_*.py' | sort | xargs -P 3 -I{} bash -c 'one {}'
