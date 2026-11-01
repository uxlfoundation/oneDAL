#!/usr/bin/env bash
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

# Action-level check of the MKL defines of DPC++ compile actions.
#
# DPC++ library objects are compiled LP64 (MKL_LP64, as Make's DPC objects are),
# so they must not carry MKL_ILP64. The DPC MKL test helper links the ILP64
# static archives (@mkl//:mkl_core), so its compile must keep MKL_ILP64, or
# its 32-bit MKL_INT calls reach 64-bit GEMM symbols.
set -euo pipefail

bazel_cmd="${BAZEL:-bazel}"
# --release_dpc=true analyses the DPC++ targets even when icpx is not in PATH.
# aquery does not run the compiler, so the missing compiler does not matter.
common=(--noshow_progress --cpu=avx2 --release_dpc=true)

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# compile_defines <label>: the -D flags of every CppCompile action of <label>.
compile_defines() {
    local label="$1"
    if ! "${bazel_cmd}" aquery "mnemonic(CppCompile, ${label})" "${common[@]}" \
        --output=text >"${work}/aquery.out" 2>"${work}/aquery.err"; then
        echo "aquery failed for ${label}" >&2
        cat "${work}/aquery.err" >&2
        exit 1
    fi
    if ! grep -q 'Mnemonic: CppCompile' "${work}/aquery.out"; then
        echo "no CppCompile action found for ${label}" >&2
        exit 1
    fi
    grep -o -- '-D[A-Z_0-9]*' "${work}/aquery.out" | sort -u
}

test_defines="$(compile_defines //cpp/oneapi/dal/test/engine/mkl:mkl_dpc)"
if ! grep -qx -- '-DMKL_ILP64' <<<"${test_defines}"; then
    echo "expected the DPC MKL test helper to compile with -DMKL_ILP64; got:" >&2
    echo "${test_defines}" >&2
    exit 1
fi

lib_defines="$(compile_defines //cpp/oneapi/dal/backend/primitives/blas:blas_dpc)"
if grep -qx -- '-DMKL_ILP64' <<<"${lib_defines}"; then
    echo "DPC library objects must be LP64 (no -DMKL_ILP64), as in Make; got:" >&2
    echo "${lib_defines}" >&2
    exit 1
fi
if ! grep -qx -- '-DMKL_LP64' <<<"${lib_defines}"; then
    echo "expected the DPC library objects to compile with -DMKL_LP64; got:" >&2
    echo "${lib_defines}" >&2
    exit 1
fi

echo "DPC MKL define checks passed"
