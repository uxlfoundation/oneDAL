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

cmake_minimum_required(VERSION 3.16)

set(ONEDAL_USE_DPCPP yes)
set(ONEDAL_INTERFACE yes)
set(ONEDAL_DISTRIBUTED yes)
set(ONEDAL_USE_CCL yes)
set(MPIEXEC_MAX_NUMPROCS "1" CACHE STRING "Number of processes")
set(MPIEXEC_NUMPROCS_PER_NODE "1" CACHE STRING "Number of processes per node")

set(MPI_C_COMPILER "mpiicx")
set(CMAKE_C_COMPILER "icx")
if(WIN32)
    set(MPI_CXX_COMPILER "mpiicx")
    set(CMAKE_CXX_COMPILER "icx")
elseif(UNIX)
    set(MPI_CXX_COMPILER "mpiicpx")
    set(CMAKE_CXX_COMPILER "icpx")
endif()

# Add cmake scripts and modules to CMake search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/")
include(setup_samples)

project(oneapi_dpc_samples)

find_package(oneDAL REQUIRED)

find_dependencies()
set_link_type()
set_common_compiler_options()

include_directories(sources)

# Initialize the EXCLUDE_LIST variable
set(EXCLUDE_LIST "sources/*.hpp")

# Define variable to specify the samples or directories to include or exclude
option(SAMPLES_LIST "")

generate_samples("${EXCLUDE_LIST}" "${SAMPLES_LIST}")
