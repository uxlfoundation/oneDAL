#!/usr/bin/env bash
set -euo pipefail
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

test -f $CONDA_PREFIX/lib/pkgconfig/dal-dynamic-threading-host.pc
test -f $CONDA_PREFIX/lib/pkgconfig/dal-static-threading-host.pc

source $CONDA_PREFIX/env/vars.sh

run_examples() {
    local interface_name=$1
    local linking_type=$2
    local extra_cmake_args=${3:-""}

    if [ "$linking_type" == "dynamic" ]; then
        library_postfix="so"
    else
        library_postfix="a"
    fi

    (
        cd examples/$interface_name/cpp
        rm -rf build_$linking_type
        mkdir -p build_$linking_type

        (
            cd build_$linking_type

            cmake_args="-DONEDAL_LINK=$linking_type"
            # Note: MKL cmake config is required for static build only and
            # doesn't work with conda-forge distribution of tbb-devel.
            # Thus, tbb is set to "found" manually.
            if [ "$linking_type" == "static" ]; then
                cmake_args="$cmake_args -DTBB_tbb_FOUND=YES"
            fi
            cmake_args="$cmake_args $extra_cmake_args"

            cmake .. $cmake_args
            make -j$(nproc)
        )

        examples_pattern="_cmake_results/intel_intel64_$library_postfix/*"
        set -- $examples_pattern
        if [ "$1" = "$examples_pattern" ]; then
            echo "ERROR: no built examples found for $interface_name-$linking_type"
            return 1
        fi

        for example in "$@"; do
            echo "================"
            echo "Running example: $interface_name-$linking_type-$(basename "$example")"
            echo "================"
            "$example"
        done
    )
}

run_dpc_examples() {
    # DPC++ examples are validated only in dynamic mode.
    # We do not produce static DPC++ artifacts in conda packaging.
    local linking_type="dynamic"
    local library_postfix="so"

    (
        cd examples/oneapi/dpc
        rm -rf build_$linking_type
        mkdir -p build_$linking_type

        (
            cd build_$linking_type

            cmake_args="-DONEDAL_LINK=$linking_type"
            # Note: MKL cmake config (incl. SYCL variant) requires TBB to be findable at configure time.
            # conda-forge tbb-devel does not ship TBBConfig.cmake, so we set TBB as found manually
            # (same approach as for the static CPU build above).
            cmake_args="$cmake_args -DTBB_tbb_FOUND=YES"

            cmake .. $cmake_args
            make -j$(nproc)
        )

        examples_pattern="_cmake_results/intel_intel64_$library_postfix/*"
        set -- $examples_pattern
        if [ "$1" = "$examples_pattern" ]; then
            echo "ERROR: no built oneapi/dpc examples found"
            return 1
        fi

        for example in "$@"; do
            echo "================"
            echo "Running example: oneapi-dpc-$linking_type-$(basename "$example")"
            echo "================"
            "$example"
        done
    )
}

# ============================================================
# CPU-only tests: oneapi/cpp + daal (no GPU lib required)
# These must pass even without dal-gpu installed.
# ============================================================
echo "========================================"
echo "Running CPU examples: oneapi/cpp dynamic"
echo "========================================"
run_examples oneapi dynamic

echo "========================================"
echo "Running CPU examples: oneapi/cpp static"
echo "========================================"
run_examples oneapi static

echo "========================================"
echo "Running CPU examples: daal/cpp dynamic"
echo "========================================"
run_examples daal dynamic

echo "========================================"
echo "Running CPU examples: daal/cpp static"
echo "========================================"
run_examples daal static

# ============================================================
# GPU/DPC++ tests: oneapi/dpc (requires dal-gpu / libonedal_dpc.so)
# Skipped if GPU library is not installed.
# ============================================================
has_dpc=$(find "$CONDA_PREFIX/lib" -maxdepth 1 -name "libonedal_dpc.so*" | head -1)
has_dpc_params=$(find "$CONDA_PREFIX/lib" -maxdepth 1 -name "libonedal_parameters_dpc.so*" | head -1)
if [ -n "$has_dpc" ] && [ -n "$has_dpc_params" ]; then
    # Workaround: MKL cmake config requires unversioned libtbb.so for tbb_thread threading.
    # conda-forge tbb-devel may only provide versioned soname (libtbb.so.<N>).
    if [ ! -f "$CONDA_PREFIX/lib/libtbb.so" ]; then
        tbb_so=$(find "$CONDA_PREFIX/lib" -maxdepth 1 -name "libtbb.so.*" | head -1)
        [ -n "$tbb_so" ] && ln -sf "$(basename "$tbb_so")" "$CONDA_PREFIX/lib/libtbb.so"
    fi
    echo "========================================"
    echo "Running GPU/DPC++ examples: oneapi/dpc dynamic"
    echo "========================================"
    run_dpc_examples
else
    echo "========================================"
    echo "Skipping GPU/DPC++ examples: dal-gpu is not fully installed"
    echo "========================================"
fi

# TODO: add testing for samples
