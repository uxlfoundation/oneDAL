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
# oneDAL 2026.1.0 CPU+GPU libs, icx/icpx 2026.1, host-side coverage (device code is not instrumentable)
set -e
U=${USAGE_MAP_WORK:-$PWD}
cd $U/oneDAL-gpu
export PATH=$U/cenv/bin:$PATH MKLROOT=$U/cenv TBBROOT=$U/cenv DPL_ROOT=$U/cenv
export TCXX_LOG=$U/compile-gpu.log; : > $TCXX_LOG
for t in daal oneapi_c oneapi_dpc; do
  make -f makefile $t PLAT=lnx32e COMPILER=icx REQCPU=avx512 CODE_COVERAGE=yes \
    "DPC_EXTRA=-Xarch_host -fprofile-arcs -Xarch_host -ftest-coverage -Wno-error=pass-failed" -j80
done
