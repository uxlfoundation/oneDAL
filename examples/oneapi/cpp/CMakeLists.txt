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

cmake_minimum_required(VERSION 3.1)

set(ONEDAL_USE_DPCPP no)
set(ONEDAL_INTERFACE yes)

# Add cmake scripts and modules to CMake search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(setup_examples)

project(oneapi_cpp_examples)

find_package(oneDAL REQUIRED)

set_link_type()
set_common_compiler_options()

include_directories(source)

option(REF_BACKEND "Exclude not supported files for non-MKL backend" OFF)
option(OPENRNG_BACKEND "Exclude not supported files for ref+OpenRNG backend" OFF)

# Initialize the EXCLUDE_LIST variable
set(EXCLUDE_LIST
    "source/utils/*"
    "source/example_util/*"
    "jaccard_batch_app"
)

if(REF_BACKEND)
    set(EXCLUDE_LIST
        ${EXCLUDE_LIST}
        "source/pca/*"
    )
    if (OPENRNG_BACKEND)
        set (EXCLUDE_LIST
            ${EXCLUDE_LIST}
            "source/decision_forest/df_reg_dense_batch.cpp"
            "source/louvain/louvain_batch.cpp"
            "source/connected_components/connected_components_batch.cpp"
        )
    endif()
endif()

include("target_excludes.cmake")

# Define variable to specify the examples or directories to include or exclude
option(EXAMPLES_LIST "")

generate_examples("${EXCLUDE_LIST}" "${EXAMPLES_LIST}")
