#===============================================================================
# Copyright 2021 Intel Corporation
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

cmake_minimum_required(VERSION 3.5)

set(ONEDAL_USE_DPCPP yes)
set(ONEDAL_INTERFACE yes)

# Add cmake scripts and modules to CMake search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(setup_examples)

project(oneapi_dpc_examples)

find_package(oneDAL REQUIRED)

set_link_type()
set_common_compiler_options()

include_directories(source)

option(ONEMATH_BACKEND "Exclude not supported files for the oneMath device backend" OFF)
option(CUDA_BACKEND "Exclude not supported files for oneMath dispatching to CUDA" OFF)

# Initialize the EXCLUDE_LIST variable
set(EXCLUDE_LIST
    "source/utils/*"
    "source/example_util/*"
)

# A library built with DPC_MATH_BACKEND=onemath / --dpc_math_backend=onemath does
# not cover every domain oneDAL uses; see the "NVIDIA GPUs through oneMath"
# section of dev/bazel/README.md. What is missing throws `unimplemented` at the
# point of use, so the examples below would build and then fail at run time --
# exclude them instead, the same way the reference host backend does.
if(ONEMATH_BACKEND)
    # The only example that hands a csr_table to the device, which is the one
    # entry into the sparse BLAS primitives.
    set(EXCLUDE_LIST
        ${EXCLUDE_LIST}
        "source/kmeans/kmeans_lloyd_csr_batch.cpp"
    )
    if (CUDA_BACKEND)
        # cuSOLVER's gesvd requires m >= n, and the PCA svd kernel calls it with
        # m = column_count, n = row_count, so it throws for any dataset with
        # more rows than columns. PCA cov goes through syevd and is unaffected.
        set(EXCLUDE_LIST
            ${EXCLUDE_LIST}
            "source/pca/pca_svd_dense_batch.cpp"
        )
    endif()
endif()

# Define variable to specify the examples or directories to include or exclude
option(EXAMPLES_LIST "")

generate_examples("${EXCLUDE_LIST}" "${EXAMPLES_LIST}")
