#!/usr/bin/env sh
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

    if [ "$linking_type" == "dynamic" ]; then
        library_postfix="so"
    else
        library_postfix="a"
    fi

    (
        cd examples/$interface_name/cpp
        mkdir build_$linking_type

        (
            cd build_$linking_type

            cmake_args="-DONEDAL_LINK=$linking_type"
            # Note: MKL cmake config is required for static build only and
            # doesn't work with conda-forge distribution of tbb-devel.
            # Thus, tbb is set to "found" manually.
            if [ "$linking_type" == "static" ]; then
                cmake_args="$cmake_args -DTBB_tbb_FOUND=YES"
            fi

            cmake .. $cmake_args
            make -j$(nproc)
        )

        for example in _cmake_results/intel_intel64_$library_postfix/*; do
            echo "================"
            echo "Running example: $interface_name-$linking_type-$(basename $example)"
            echo "================"
            $example
        done
    )
}

run_samples() {
    local interface_name=$1      # e.g. cxx or fortran
    local linking_type=$2        # dynamic or static
    local device_type=$3         # cpu, gpu, etc.
    local communicator_type=$4   # mpi, serial, etc.

    echo "========================================================"
    echo "STARTING SAMPLE BUILD AND RUN"
    echo "Parameters:"
    echo "  interface_name      = $interface_name"
    echo "  linking_type        = $linking_type"
    echo "  device_type         = $device_type"
    echo "  communicator_type   = $communicator_type"
    echo "========================================================"

    # Determine library extension
    if [ "$linking_type" == "dynamic" ]; then
        library_postfix="so"
        echo "→ Linking type: dynamic → looking for .so libraries"
    else
        library_postfix="a"
        echo "→ Linking type: static → looking for .a libraries"
    fi

    local samples_dir="samples/$interface_name/$device_type/$communicator_type"
    echo "Changing to samples directory: $samples_dir"

    if [ ! -d "$samples_dir" ]; then
        echo "ERROR: Directory $samples_dir does NOT exist!"
        echo "Content of samples/$interface_name/:"
        ls -la "samples/$interface_name/" 2>/dev/null || echo "  (nothing found)"
        return 1
    fi

    echo "Content of $samples_dir:"
    ls -la "$samples_dir"

    (
        cd "$samples_dir" || { echo "Failed to enter $samples_dir"; exit 1; }

        echo "Current directory: $(pwd)"
        echo "Creating build directory: build_$linking_type"
        mkdir -p "build_$linking_type"

        (
            cd "build_$linking_type" || { echo "Failed to enter build_$linking_type"; exit 1; }

            echo "Now in: $(pwd)"
            echo "Cleaning previous build (just in case)..."
            rm -rf *

            local cmake_args="-DONEDAL_LINK=$linking_type"

            if [ "$linking_type" == "static" ]; then
                echo "→ Static build: forcing CMake to treat TBB as found"
                cmake_args="$cmake_args -DTBB_tbb_FOUND=YES"
            fi

            echo "Running CMake with arguments: $cmake_args"
            echo "Full command: cmake .. $cmake_args"
            cmake .. $cmake_args
            if [ $? -ne 0 ]; then
                echo "CMake FAILED!"
                exit 1
            fi

            echo "CMake completed successfully."
            echo "Content of build directory after CMake:"
            ls -la

            echo "Starting compilation: make -j$(nproc)"
            make -j$(nproc)
            if [ $? -ne 0 ]; then
                echo "make FAILED!"
                exit 1
            fi

            echo "Build completed successfully!"
            echo "Content of build directory after make:"
            ls -la
        )

        # Look for compiled samples
        local results_dir="_cmake_results/intel_intel64_$library_postfix"
        echo "Searching for compiled samples in: $results_dir"

        if [ ! -d "$results_dir" ]; then
            echo "ERROR: Results directory not found: $results_dir"
            echo "Current directory content:"
            ls -la
            return 1
        fi

        echo "Found results directory: $results_dir"
        echo "Directory content:"
        ls -la "$results_dir"

        local count=0
        for sample in "$results_dir"/*; do
            # Skip if glob didn't match anything
            [ -f "$sample" ] || continue
            [ -x "$sample" ] || { echo "WARNING: $sample is not executable"; continue; }

            count=$((count + 1))
            echo ""
            echo "========================================================"
            echo "RUNNING SAMPLE #$count"
            echo "Name: $interface_name-$linking_type-$(basename "$sample")"
            echo "Full path: $sample"
            echo "========================================================"

            mpirun -n 2 "$sample"
            echo "Sample finished with exit code: $?"
            echo "========================================================"
            echo ""
        done

        if [ $count -eq 0 ]; then
            echo "WARNING: No executable samples found in $results_dir"
        else
            echo "All done! Successfully ran $count sample(s)."
        fi
    )
}

run_examples oneapi dynamic
# run_examples oneapi static
# run_examples daal dynamic
# run_examples daal static
run_samples oneapi dynamic cpp mpi
# run_samples oneapi static cpp mpi
