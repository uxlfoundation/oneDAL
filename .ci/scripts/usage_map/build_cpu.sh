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
# $1 = rel (O2, all ISAs, timed: cost) | cov2 (O0, sse2 avx2 avx512, coverage + callgraph)
set -e
D=$(cd "$(dirname "$0")" && pwd)
U=${USAGE_MAP_WORK:-$PWD}
K=$1; cd $U/oneDAL-$K
export MKLROOT=${MKL_PREFIX:?set MKL_PREFIX to an MKL install} TBBROOT=$PWD/__deps/tbb/lnx
export TCXX_LOG=$U/compile-$K.log; : > $TCXX_LOG
if [ $K = cov2 ]; then
  export CXXFLAGS="--coverage -fprofile-update=atomic -fcallgraph-info" LDFLAGS=--coverage
  X="OPTFLAG=O0 REQCPU=sse2 avx2 avx512"
  for t in daal oneapi_c; do make -f makefile $t PLAT=lnx32e COMPILER=gnu CXX=$D/tcxx OPTFLAG=O0 "REQCPU=sse2 avx2 avx512" -j${JOBS:-32}; done
else
  for t in daal oneapi_c; do make -f makefile $t PLAT=lnx32e COMPILER=gnu CXX=$D/tcxx -j${JOBS:-32}; done
fi
