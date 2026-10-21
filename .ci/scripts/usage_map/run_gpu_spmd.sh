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
# GPU SPMD unit: sklearnex/onedal spmd tests on 4 MPI ranks over host-instrumented oneDAL-gpu
D=$(cd "$(dirname "$0")" && pwd)
U=${USAGE_MAP_WORK:-$PWD}
S=$U/sklearnex-gpusrc
export OD=oneDAL-gpu I_MPI_FABRICS=shm
/usr/bin/time -f "spmd %e s" env PATH=$U/genv/bin:$PATH $U/genv/bin/mpirun -n 4 $D/run_gpu.sh gpu_spmd $S/tests/helper_mpi_tests.py \
  pytest -q --with-mpi --timeout=1800 -c $S/setup.cfg $S/onedal/spmd $S/sklearnex/spmd > $U/gpu-tests-spmd.log 2>&1
grep -E "^=.*(passed|failed|skipped)" $U/gpu-tests-spmd.log | tail -1
