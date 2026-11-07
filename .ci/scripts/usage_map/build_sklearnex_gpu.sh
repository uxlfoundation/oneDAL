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
# sklearnex 2026.1 extensions (incl. SPMD, Intel MPI from mpienv) from source over oneDAL-gpu (same compiler, consistent ABI)
set -e
U=${USAGE_MAP_WORK:-$PWD}
cd $U/sklearnex-gpusrc
export PATH=$U/genv/bin:$U/cenv/bin:$PATH DALROOT=$U/oneDAL-gpu/__release_lnx/daal/latest
export MKLROOT=$U/cenv TBBROOT=$U/cenv MPIROOT=$U/mpienv CC=icx CXX=icpx
python setup.py build_ext --inplace -j 32
python setup.py build -j 32
cp -v build/lib*/onedal/*.so onedal/
