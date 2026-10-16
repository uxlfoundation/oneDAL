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

# An EXE and a DLL that link oneDAL statically; check_windows_static_exports.sh inspects them.
cmake_minimum_required(VERSION 3.16)
project(windows_static_exports CXX)

set(ONEDAL_LINK static)
set(ONEDAL_USE_DPCPP no)
set(ONEDAL_INTERFACE yes)
find_package(oneDAL REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE ${oneDAL_IMPORTED_TARGETS})

add_library(plugin SHARED plugin.cpp)
target_link_libraries(plugin PRIVATE ${oneDAL_IMPORTED_TARGETS})

add_executable(plugin_host plugin_host.cpp)
target_link_libraries(plugin_host PRIVATE plugin)

# The link maps show which oneDAL functions the linker kept
foreach(target consumer plugin)
    target_link_options(${target} PRIVATE "/MAP:$<TARGET_FILE_DIR:${target}>/${target}.map")
endforeach()
