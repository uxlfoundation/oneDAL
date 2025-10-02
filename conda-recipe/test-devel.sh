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

test -f $PREFIX/lib/pkgconfig/dal-dynamic-threading-host.pc
test -f $PREFIX/lib/pkgconfig/dal-static-threading-host.pc

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

run_examples oneapi dynamic
run_examples oneapi static
run_examples daal dynamic
run_examples daal static
