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
# GPU units: onedal and sklearnex pytest from sklearnex-gpusrc over host-instrumented oneDAL-gpu (coverage build)
D=$(cd "$(dirname "$0")" && pwd)
U=${USAGE_MAP_WORK:-$PWD}
for p in onedal sklearnex; do
  /usr/bin/time -f "$p %e s" env OD=oneDAL-gpu $D/run_gpu.sh u/gpu_$p -m pytest -q -n 4 --timeout=1800 \
    -c $U/sklearnex-gpusrc/setup.cfg $U/sklearnex-gpusrc/$p > $U/gpu-tests-$p.log 2>&1
  tail -1 $U/gpu-tests-$p.log
done
