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
#
# Builds the abicheck-facts Clang plugin (libabicheck-facts.so) against the
# LLVM/Clang toolchain bundled inside the Intel oneAPI DPC++/C++ Compiler
# (icx/icpx), so the plugin loads correctly into icx itself.
#
# icx is a clang-based compiler: the intel-oneapi-compiler-dpcpp-cpp package
# installs its own LLVM/Clang CMake package config under
# $CMPLR_ROOT/lib/cmake/{llvm,clang}. Building the plugin against that
# CMAKE_PREFIX_PATH (rather than a separately apt-installed upstream clang)
# guarantees the plugin links against the exact LLVM major/ABI that icx loads
# at compile time -- there is no separate "which LLVM major does icx use"
# guess involved.
#
# Usage: abicheck_build_plugin.sh <output-dir>

set -eo pipefail

out_dir=${1:?usage: abicheck_build_plugin.sh <output-dir>}
plugin_ref="v0.5.0"
plugin_repo="https://github.com/abicheck/abicheck.git"

if [ -z "${CMPLR_ROOT}" ]; then
    echo "::error:: CMPLR_ROOT is not set. Source /opt/intel/oneapi/setvars.sh before calling this script."
    exit 1
fi

llvm_cmake_dir="${CMPLR_ROOT}/lib/cmake/llvm"
clang_cmake_dir="${CMPLR_ROOT}/lib/cmake/clang"
if [ ! -d "${llvm_cmake_dir}" ] || [ ! -d "${clang_cmake_dir}" ]; then
    echo "::error:: Could not find LLVM/Clang CMake package config under \$CMPLR_ROOT ($CMPLR_ROOT)."
    echo "Expected: ${llvm_cmake_dir} and ${clang_cmake_dir}"
    exit 1
fi

work_dir=$(mktemp -d)
trap 'rm -rf "${work_dir}"' EXIT

git clone --depth 1 --branch "${plugin_ref}" "${plugin_repo}" "${work_dir}/abicheck"

cmake -S "${work_dir}/abicheck/contrib/abicheck-clang-plugin" -B "${work_dir}/build" -G Ninja \
    -DCMAKE_PREFIX_PATH="${CMPLR_ROOT}" \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx
cmake --build "${work_dir}/build"

mkdir -p "${out_dir}"
cp "${work_dir}/build/libabicheck-facts.so" "${out_dir}/libabicheck-facts.so"

echo "Built ${out_dir}/libabicheck-facts.so against icx's bundled LLVM/Clang (${CMPLR_ROOT})."
