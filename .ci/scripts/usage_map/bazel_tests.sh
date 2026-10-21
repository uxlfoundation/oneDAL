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
# oneDAL's own Bazel tests (//cpp/oneapi/dal:tests) on the cov2 commit with gcc --coverage.
# gcda go to cov-cov2/u/bazel_onedal-tests via GCOV_PREFIX; never set MKLROOT/TBBROOT here.
set -x
U=${USAGE_MAP_WORK:-$PWD}
unset MKLROOT TBBROOT
export CC=gcc
mkdir -p $U/cov-cov2/u/bazel_onedal-tests
cd $U/oneDAL-bzt
time ${BAZEL:-bazel} --output_user_root=$U/bzroot test //cpp/oneapi/dal:tests \
  --cpu=avx2 --copt=--coverage --copt=-O0 --linkopt=--coverage --spawn_strategy=local \
  --jobs=48 --local_test_jobs=16 --test_output=errors --keep_going \
  --test_env=GCOV_PREFIX=$U/cov-cov2/u/bazel_onedal-tests --test_env=GCOV_PREFIX_STRIP=0
