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

# sycl_cache.sh — Enable SYCL persistent JIT cache and warm it up by running
# oneDAL DPC examples from a ready release.
#
# This script is deployed alongside vars.sh in the env/ directory of a oneDAL
# release (e.g. __release_lnx/daal/latest/env/sycl_cache.sh).
#
# Usage:
#   __release_lnx/daal/latest/env/sycl_cache.sh [options]
#
# The script:
#   1. Configures SYCL_CACHE_PERSISTENT and related env vars
#   2. Sources vars.sh to set up the oneDAL environment
#   3. Builds DPC examples (cmake + make) from the release
#   4. Runs every built example so the SYCL runtime caches JIT-compiled binaries
#   5. On subsequent runs with the same cache dir, JIT compilation is skipped
#
# The cache directory is reusable across runs. Point SYCL_CACHE_DIR at it
# before launching any SYCL workload to benefit from pre-warmed kernels.

set -eo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
# env/ lives under daal/latest/ — go one level up to get DAL_ROOT
DAL_ROOT=$(readlink -f "${SCRIPT_DIR}/..")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SYCL_CACHE_DIR_DEFAULT="${DAL_ROOT}/__sycl_cache"
CACHE_MAX_SIZE_MB=8192          # 8 GB
CACHE_THRESHOLD_DAYS=30         # evict after 30 days
WARMUP=yes                      # run examples to populate cache
COMPILER=icx
JOBS=""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [options]"
    column -t -s":" <<< '--help:Display this information
--cache-dir <path>:SYCL persistent cache directory (default: <DAL_ROOT>/__sycl_cache)
--cache-max-size <MB>:Max cache size in MB (default: 8192)
--cache-threshold <days>:Eviction threshold in days (default: 30)
--compiler <name>:Compiler to build examples with (default: icx)
--no-warmup:Only export env vars, skip building/running examples
--jobs <N>:Number of parallel build jobs
'
}

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cache-dir)    SYCL_CACHE_DIR_USER="$2"; shift ;;
        --cache-max-size) CACHE_MAX_SIZE_MB="$2"; shift ;;
        --cache-threshold) CACHE_THRESHOLD_DAYS="$2"; shift ;;
        --compiler)     COMPILER="$2";           shift ;;
        --no-warmup)    WARMUP=no ;;
        --jobs)         JOBS="$2";               shift ;;
        --help)         show_help; exit 0 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# 1. Configure SYCL persistent cache
# ---------------------------------------------------------------------------
export SYCL_CACHE_DIR="${SYCL_CACHE_DIR_USER:-${SYCL_CACHE_DIR_DEFAULT}}"
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_MAX_SIZE="${CACHE_MAX_SIZE_MB}"
export SYCL_CACHE_THRESHOLD="${CACHE_THRESHOLD_DAYS}"
export SYCL_CACHE_IN_MEM=1

mkdir -p "${SYCL_CACHE_DIR}"

echo "============================================"
echo " SYCL Persistent Cache Configuration"
echo "============================================"
echo " SYCL_CACHE_PERSISTENT = ${SYCL_CACHE_PERSISTENT}"
echo " SYCL_CACHE_DIR        = ${SYCL_CACHE_DIR}"
echo " SYCL_CACHE_MAX_SIZE   = ${SYCL_CACHE_MAX_SIZE} MB"
echo " SYCL_CACHE_THRESHOLD  = ${SYCL_CACHE_THRESHOLD} days"
echo " SYCL_CACHE_IN_MEM     = ${SYCL_CACHE_IN_MEM}"
echo "============================================"

if [ "${WARMUP}" != "yes" ]; then
    echo "Cache env vars exported. Skipping warmup (--no-warmup)."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Set up oneDAL + TBB environment
# ---------------------------------------------------------------------------
DAL_ENV="${SCRIPT_DIR}/vars.sh"
if [ ! -f "${DAL_ENV}" ]; then
    echo "ERROR: Cannot find vars.sh at ${DAL_ENV}"
    echo "       sycl_cache.sh must sit in the env/ directory next to vars.sh."
    exit 1
fi

# vars.sh must be sourced (not executed)
# shellcheck disable=SC1090
source "${DAL_ENV}"

# TBB: check TBBROOT, or try well-known locations relative to DAL_ROOT
if [ -z "${TBBROOT:-}" ]; then
    REPO_ROOT=$(readlink -f "${DAL_ROOT}/../..")
    for tbb_candidate in \
        "${REPO_ROOT}/__deps/tbb/lnx" \
        "${REPO_ROOT}/__deps/tbb-$(uname -m)" \
        "${REPO_ROOT}/__deps/tbb"; do
        if [ -d "${tbb_candidate}" ]; then
            export TBBROOT="${tbb_candidate}"
            break
        fi
    done
fi

if [ -n "${TBBROOT:-}" ]; then
    export CPATH="${TBBROOT}/include${CPATH:+:$CPATH}"
    export CMAKE_PREFIX_PATH="${TBBROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    if [ -d "${TBBROOT}/lib/intel64/gcc4.8" ]; then
        TBB_LIB_DIR="${TBBROOT}/lib/intel64/gcc4.8"
    else
        TBB_LIB_DIR="${TBBROOT}/lib"
    fi
    export LD_LIBRARY_PATH="${TBB_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${TBB_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    echo " TBBROOT = ${TBBROOT}"
else
    echo "WARNING: TBBROOT not found — TBB must already be in your environment"
fi

# ---------------------------------------------------------------------------
# 3. Determine compiler
# ---------------------------------------------------------------------------
if [[ "${COMPILER}" == "icx" ]]; then
    export CC=icx
    export CXX=icpx
elif [[ "${COMPILER}" == "gnu" ]]; then
    export CC=gcc
    export CXX=g++
elif [[ "${COMPILER}" == "clang" ]]; then
    export CC=clang
    export CXX=clang++
else
    echo "ERROR: Unsupported compiler '${COMPILER}'"
    exit 1
fi

if [[ -n "${JOBS}" ]]; then
    MAKE_J="-j${JOBS}"
else
    MAKE_J="-j$(nproc --all)"
fi

# ---------------------------------------------------------------------------
# 4. Build & run DPC examples to warm up SYCL cache
# ---------------------------------------------------------------------------
# Only DPC examples use SYCL JIT — CPU-only interfaces (oneapi/cpp, daal/cpp)
# do not generate device code, so there is nothing to cache.
# Within DPC, we run one example per algorithm directory since different
# variants (batch/online, dense/csr, etc.) compile the same SYCL kernels.

WARMUP_RETURN=0
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0
TOTAL_DEDUP=0

EXAMPLES_BASE="${DAL_ROOT}/examples"

# Source directories to skip (not real algorithms)
SKIP_DIRS="example_util misc table"

build_and_run_examples() {
    local examples_dir="${EXAMPLES_BASE}/oneapi/dpc"

    if [ ! -f "${examples_dir}/CMakeLists.txt" ]; then
        echo "  [SKIP] oneapi/dpc — no CMakeLists.txt"
        return
    fi

    echo ""
    echo "--------------------------------------------"
    echo " oneapi/dpc (one example per algorithm)"
    echo " dir: ${examples_dir}"
    echo " compiler: ${COMPILER} (CC=${CC}, CXX=${CXX})"
    echo "--------------------------------------------"

    pushd "${examples_dir}" > /dev/null

    # Clean previous build artifacts
    rm -rf Build _cmake_results

    cmake -B Build -S . \
        -G "Unix Makefiles" \
        -DONEDAL_LINK="dynamic" 2>&1 | tail -5

    make ${MAKE_J} -C Build 2>&1 | tail -3
    local err=$?
    if [ ${err} -ne 0 ]; then
        echo "  ERROR: Build failed (exit code ${err})"
        WARMUP_RETURN=${err}
        popd > /dev/null
        return
    fi

    # Find built example binaries
    local results_dir
    results_dir=$(find _cmake_results -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
    if [ -z "${results_dir}" ]; then
        results_dir="Build"
    fi

    # Match each binary to its algorithm source directory and deduplicate
    local src_dir="${examples_dir}/source"
    declare -A seen_algos

    for example_bin in "${results_dir}"/*; do
        [ -f "${example_bin}" ] || continue
        [ -x "${example_bin}" ] || continue

        local example_name
        example_name=$(basename "${example_bin}")

        # Find which algorithm directory this example belongs to
        local algo=""
        for algo_dir in "${src_dir}"/*/; do
            local algo_name
            algo_name=$(basename "${algo_dir}")
            case " ${SKIP_DIRS} " in *" ${algo_name} "*) continue ;; esac
            if [ -f "${algo_dir}/${example_name}.cpp" ]; then
                algo="${algo_name}"
                break
            fi
        done

        # Skip non-algorithm examples
        if [ -z "${algo}" ]; then
            continue
        fi

        # One example per algorithm is enough
        if [ -n "${seen_algos[${algo}]+x}" ]; then
            echo "  [DEDUP] ${example_name} — ${algo} already cached"
            TOTAL_DEDUP=$((TOTAL_DEDUP + 1))
            continue
        fi
        seen_algos[${algo}]=1

        echo -n "  [WARMUP] ${example_name} ... "

        # Run with a timeout to avoid hangs
        if timeout 120 "${example_bin}" > /dev/null 2>&1; then
            echo "OK"
            TOTAL_PASS=$((TOTAL_PASS + 1))
        else
            local exit_code=$?
            if [ ${exit_code} -eq 124 ]; then
                echo "TIMEOUT (skipped)"
                TOTAL_SKIP=$((TOTAL_SKIP + 1))
            else
                echo "FAILED (exit ${exit_code})"
                TOTAL_FAIL=$((TOTAL_FAIL + 1))
                WARMUP_RETURN=${exit_code}
            fi
        fi
    done

    # Cleanup build artifacts (keep the cache, not the build)
    rm -rf Build _cmake_results
    popd > /dev/null
}

echo ""
echo "============================================"
echo " Building & running DPC examples for cache warmup"
echo "============================================"

build_and_run_examples

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
CACHE_SIZE=$(du -sh "${SYCL_CACHE_DIR}" 2>/dev/null | cut -f1)

echo ""
echo "============================================"
echo " SYCL Cache Warmup Complete"
echo "============================================"
echo " Passed  : ${TOTAL_PASS}"
echo " Skipped : ${TOTAL_DEDUP} (deduplicated)"
echo " Failed  : ${TOTAL_FAIL}"
echo " Timeout : ${TOTAL_SKIP}"
echo " Cache   : ${SYCL_CACHE_DIR} (${CACHE_SIZE})"
echo "============================================"
echo ""
echo "To reuse this cache in future runs, export:"
echo ""
echo "  export SYCL_CACHE_PERSISTENT=1"
echo "  export SYCL_CACHE_DIR=${SYCL_CACHE_DIR}"
echo ""

exit ${WARMUP_RETURN}
