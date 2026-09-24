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
# Usage (source to keep env vars in your shell):
#   source __release_lnx/daal/latest/env/sycl_cache.sh [options]
#
# Or run as a standalone script (env vars stay in subprocess only):
#   __release_lnx/daal/latest/env/sycl_cache.sh [options]
#
# The script:
#   1. Configures SYCL_CACHE_PERSISTENT and related env vars
#   2. Sources vars.sh to set up the oneDAL environment
#   3. Builds DPC examples (cmake + make) from the release
#   4. Runs every built example on GPU so the SYCL runtime caches
#      JIT-compiled native binaries from SPIR-V device code
#   5. On subsequent runs with the same cache dir, JIT compilation is skipped
#
# GPU vs CPU:
#   The SYCL persistent cache benefits GPU (Level-Zero) workloads where
#   SPIR-V → native ISA JIT compilation causes 1.5-6x overhead on truly
#   first invocation (before any driver or runtime cache exists).
#   CPU (OpenCL) does not produce cacheable kernel binaries — the warmup
#   phase targets GPU devices only.
#
# Interaction with L0 driver cache:
#   Intel's Level-Zero driver maintains its own cache at
#   ~/.cache/neo_compiler_cache/.  When that cache is warm, the SYCL
#   persistent cache adds little.  However, the SYCL cache acts as a
#   portable safety net: it is user-controlled, survives driver upgrades,
#   and ships with the oneDAL release so first-time users on a clean
#   machine get cached kernels immediately after warmup.
#
# The cache directory is reusable across runs. Point SYCL_CACHE_DIR at it
# before launching any SYCL workload to benefit from pre-warmed kernels.

# ---------------------------------------------------------------------------
# Wrap everything in a main function so `return` works when sourced.
# When sourced, `return` from _sycl_cache_main exits the function (and thus
# the script) without killing the caller's shell.
# ---------------------------------------------------------------------------
_sycl_cache_main() {

local _sycl_cache_sourced=false
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _sycl_cache_sourced=true
fi

local SCRIPT_PATH SCRIPT_DIR DAL_ROOT
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
# env/ lives under daal/latest/ — go one level up to get DAL_ROOT
DAL_ROOT=$(readlink -f "${SCRIPT_DIR}/..")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
local SYCL_CACHE_DIR_DEFAULT="${DAL_ROOT}/__sycl_cache"
local CACHE_MAX_SIZE_MB=8192          # 8 GB
local CACHE_THRESHOLD_DAYS=30         # evict after 30 days
local WARMUP=yes                      # run examples to populate cache
local COMPILER=icx
local JOBS=""
local SYCL_CACHE_DIR_USER=""

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    local key="$1"
    case $key in
        --cache-dir)    SYCL_CACHE_DIR_USER="$2"; shift ;;
        --cache-max-size) CACHE_MAX_SIZE_MB="$2"; shift ;;
        --cache-threshold) CACHE_THRESHOLD_DAYS="$2"; shift ;;
        --compiler)     COMPILER="$2";           shift ;;
        --no-warmup)    WARMUP=no ;;
        --jobs)         JOBS="$2";               shift ;;
        --help)
            echo "Usage: sycl_cache.sh [options]"
            column -t -s":" <<< '--help:Display this information
--cache-dir <path>:SYCL persistent cache directory (default: <DAL_ROOT>/__sycl_cache)
--cache-max-size <MB>:Max cache size in MB (default: 8192)
--cache-threshold <days>:Eviction threshold in days (default: 30)
--compiler <name>:Compiler to build examples with (default: icx)
--no-warmup:Only export env vars, skip building/running examples
--jobs <N>:Number of parallel build jobs
'
            return 0 ;;
        *)              echo "Unknown option: $1"; return 1 ;;
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
    return 0
fi

# ---------------------------------------------------------------------------
# 2. Set up oneDAL + TBB environment
# ---------------------------------------------------------------------------
local DAL_ENV="${SCRIPT_DIR}/vars.sh"
if [ ! -f "${DAL_ENV}" ]; then
    echo "ERROR: Cannot find vars.sh at ${DAL_ENV}"
    echo "       sycl_cache.sh must sit in the env/ directory next to vars.sh."
    return 1
fi

# vars.sh must be sourced (not executed)
# shellcheck disable=SC1090
source "${DAL_ENV}"

# TBB: check TBBROOT, or try well-known locations relative to DAL_ROOT
if [ -z "${TBBROOT:-}" ]; then
    local REPO_ROOT
    REPO_ROOT=$(readlink -f "${DAL_ROOT}/../..")
    local tbb_candidate
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
    local TBB_LIB_DIR
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
    return 1
fi

local MAKE_J
if [[ -n "${JOBS}" ]]; then
    MAKE_J="-j${JOBS}"
else
    MAKE_J="-j$(nproc --all)"
fi

# ---------------------------------------------------------------------------
# 4. Build & run DPC examples to warm up SYCL cache (GPU only)
# ---------------------------------------------------------------------------
# Only DPC examples use SYCL JIT — CPU-only interfaces (oneapi/cpp, daal/cpp)
# do not generate device code, so there is nothing to cache.
#
# CPU (OpenCL) note: benchmarks show that CPU OpenCL produces 0 cached kernel
# binaries — the SYCL persistent cache mechanism only benefits GPU (Level-Zero)
# where SPIR-V → native ISA JIT adds 1.5-6x overhead on first invocation
# (when no driver cache exists).
# The warmup phase therefore targets GPU devices only via ONEAPI_DEVICE_SELECTOR.
#
# Within DPC, we run one example per algorithm directory since different
# variants (batch/online, dense/csr, etc.) compile the same SYCL kernels.

local WARMUP_RETURN=0
local TOTAL_PASS=0
local TOTAL_FAIL=0
local TOTAL_SKIP=0
local TOTAL_DEDUP=0

local EXAMPLES_BASE="${DAL_ROOT}/examples"

# Source directories to skip (not real algorithms)
local SKIP_DIRS="example_util misc table"

local examples_dir="${EXAMPLES_BASE}/oneapi/dpc"

if [ ! -f "${examples_dir}/CMakeLists.txt" ]; then
    echo "  [SKIP] oneapi/dpc — no CMakeLists.txt"
    return 0
fi

echo ""
echo "============================================"
echo " Building & running DPC examples for GPU cache warmup"
echo "============================================"
echo ""
echo "--------------------------------------------"
echo " oneapi/dpc (one example per algorithm, GPU only)"
echo " dir: ${examples_dir}"
echo " compiler: ${COMPILER} (CC=${CC}, CXX=${CXX})"
echo "--------------------------------------------"

pushd "${examples_dir}" > /dev/null

# Clean previous build artifacts
rm -rf Build _cmake_results

cmake -B Build -S . \
    -G "Unix Makefiles" \
    -DONEDAL_LINK="dynamic" 2>&1 | tail -5

# Allow partial build success — some examples may fail to link (e.g. WIP
# algorithms) while the rest compile fine.  We still want to run whatever
# was built to populate the SYCL cache.
make ${MAKE_J} -C Build 2>&1 | tail -5 || true

# Find built example binaries
local results_dir
results_dir=$(find _cmake_results -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
if [ -z "${results_dir}" ]; then
    results_dir="Build"
fi

# Match each binary to its algorithm source directory and deduplicate
local src_dir="${examples_dir}/source"
declare -A seen_algos

# Target GPU only — CPU OpenCL does not benefit from persistent cache
local saved_device_selector="${ONEAPI_DEVICE_SELECTOR:-}"
export ONEAPI_DEVICE_SELECTOR="level_zero:*"

local example_bin example_name algo algo_dir algo_name
local start_ns end_ns elapsed_ms exit_code

for example_bin in "${results_dir}"/*; do
    [ -f "${example_bin}" ] || continue
    [ -x "${example_bin}" ] || continue

    example_name=$(basename "${example_bin}")

    # Find which algorithm directory this example belongs to
    algo=""
    for algo_dir in "${src_dir}"/*/; do
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

    # Time each warmup to show JIT overhead being captured
    start_ns=$(date +%s%N)

    echo -n "  [WARMUP] ${example_name} ... "

    # Run with a timeout to avoid hangs
    if timeout 120 "${example_bin}" > /dev/null 2>&1; then
        end_ns=$(date +%s%N)
        elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
        echo "OK (${elapsed_ms} ms)"
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        exit_code=$?
        end_ns=$(date +%s%N)
        elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
        if [ ${exit_code} -eq 124 ]; then
            echo "TIMEOUT (${elapsed_ms} ms, skipped)"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
        else
            echo "FAILED (exit ${exit_code}, ${elapsed_ms} ms)"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            WARMUP_RETURN=${exit_code}
        fi
    fi
done

# Restore device selector
if [ -n "${saved_device_selector}" ]; then
    export ONEAPI_DEVICE_SELECTOR="${saved_device_selector}"
else
    unset ONEAPI_DEVICE_SELECTOR
fi

# Cleanup build artifacts (keep the cache, not the build)
rm -rf Build _cmake_results
popd > /dev/null

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
local CACHE_SIZE CACHE_BINS
CACHE_SIZE=$(du -sh "${SYCL_CACHE_DIR}" 2>/dev/null | cut -f1)
CACHE_BINS=$(find "${SYCL_CACHE_DIR}" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)

echo ""
echo "============================================"
echo " SYCL Cache Warmup Complete"
echo "============================================"
echo " Passed  : ${TOTAL_PASS}"
echo " Skipped : ${TOTAL_DEDUP} (deduplicated)"
echo " Failed  : ${TOTAL_FAIL}"
echo " Timeout : ${TOTAL_SKIP}"
echo " Kernels : ${CACHE_BINS} cached entries"
echo " Cache   : ${SYCL_CACHE_DIR} (${CACHE_SIZE})"
echo "============================================"
echo ""
if $_sycl_cache_sourced; then
    echo "Cache env vars are now active in your shell."
    echo "All subsequent GPU (Level-Zero) SYCL workloads will"
    echo "skip JIT compilation for cached kernels (1.5-6x faster"
    echo "first invocation on a clean machine)."
else
    echo "To reuse this cache in future sessions, export:"
    echo ""
    echo "  export SYCL_CACHE_PERSISTENT=1"
    echo "  export SYCL_CACHE_DIR=${SYCL_CACHE_DIR}"
fi
echo ""

return ${WARMUP_RETURN}
}

# Run main and propagate exit code
_sycl_cache_main "$@"
