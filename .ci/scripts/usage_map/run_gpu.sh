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
# sklearnex-gpusrc over oneDAL-$OD libs with one consistent 2026.1 SYCL/MKL stack (menv), genv only for python/dpctl
U=${USAGE_MAP_WORK:-$PWD}; M=$U/menv/lib; G=$U/genv/lib; L=$U/${OD:-oneDAL-gpu-m}/__release_lnx/daal/latest/lib/intel64
export GCOV_PREFIX=$U/cov-gpu/$1 GCOV_PREFIX_STRIP=0 SCIPY_ARRAY_API=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4
mkdir -p $GCOV_PREFIX; export LD_LIBRARY_PATH=$M:$U/cenv/lib:$G
export LD_PRELOAD=$M/libsycl.so.9:$M/libmkl_core.so.3:$M/libmkl_intel_lp64.so.3:$M/libmkl_tbb_thread.so.3:$L/libonedal_core.so.4:$L/libonedal_thread.so.4:$L/libonedal_dpc.so.4:$L/libonedal_parameters_dpc.so.4
export PYTHONPATH=$U/sklearnex-gpusrc
shift; mkdir -p "$U/run-u" && cd "$U/run-u" || exit 1
exec $U/genv/bin/python "$@"
