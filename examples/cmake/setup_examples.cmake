#===============================================================================
# Copyright 2023 Intel Corporation
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

include_guard()

#Defines mapping between link mode and filenames
function (set_link_type)
    if ("${ONEDAL_LINK}" STREQUAL "static")
        set(LINK_TYPE "a" PARENT_SCOPE)
    else()
        set(LINK_TYPE "so" PARENT_SCOPE)
    endif()
endfunction()

function (set_common_compiler_options)
    #Setting base common set of params for examples compilation
    if(WIN32)
        add_compile_options(/W3 /EHsc)
    elseif(APPLE)
        add_compile_options(-pedantic -Wall -Wextra -Werror -Wno-unused-parameter)
    elseif(UNIX)
        add_compile_options(-pedantic -Wall -Wextra -Werror -Wno-unused-parameter)
    endif()
    #For DAAL interfaces remove deprecated warnings
    if (ONEDAL_INTERFACE STREQUAL "no")
        if (UNIX)
            add_compile_options(-Wno-deprecated-declarations)
        elseif(WIN32)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4078")
        endif()
        if ("${CMAKE_C_COMPILER}" STREQUAL "gcc")
            add_compile_options(-std=c++03 -Wno-variadic-macros -Wno-long-long)
        endif()
    endif()
    if(CMAKE_CXX_COMPILER MATCHES "^icp?x$|^.*/(icpx|icx\.exe)$")
        if(WIN32)
            add_compile_options(-Wall -w)
        endif()
    endif()
    # Disable icc depreacation warnings
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if(WIN32)
            set(DIAG_DISABLE "/Qdiag-disable:10441")
        else()
            set(DIAG_DISABLE "-diag-disable=10441")
        endif()
        add_compile_options(${DIAG_DISABLE})
        add_link_options(${DIAG_DISABLE})
    endif()
    if(ONEDAL_USE_DPCPP STREQUAL "yes" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # link huge device code for DPCPP
        # without this flag build fails with relocation errors
        if(WIN32)
            add_link_options("/flink-huge-device-code")
        else()
            add_link_options("-flink-huge-device-code")
        endif()
    endif()
    message(STATUS "Common compiler params set")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
    message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
    message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
endfunction()


#Funtion for resetting mode to MDD on windows
function (change_md_to_mdd)
    set(cxx_flag ${CMAKE_CXX_FLAGS})
    set(cxxr_flag ${CMAKE_CXX_FLAGS_RELEASE})
    set(c_flag ${CMAKE_C_FLAGS})
    set(cr_flag ${CMAKE_C_FLAGS_RELEASE})
    set(flags
            cxx_flag
            cxxr_flag
            c_flag
            cr_flag)
    foreach(flag ${flags})
        string(REPLACE "/MD" "/MDd /debug:none" ${flag} "${${flag}}")
    endforeach()

    set(CMAKE_CXX_FLAGS ${cxx_flag} PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS_RELEASE ${cxxr_flag} PARENT_SCOPE)
    set(CMAKE_C_FLAGS ${c_flag} PARENT_SCOPE)
    set(CMAKE_C_FLAGS_RELEASE ${cr_flag} PARENT_SCOPE)
endfunction()

#Function for adding new examples to CMAKE configuration based on list of examples paths
function (add_examples examples_paths)
    foreach(example_file_path ${examples_paths})
        get_filename_component(example ${example_file_path} NAME_WE)

        # Detect CPU architecture
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
            set(CPU_ARCHITECTURE "intel_intel64")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(CPU_ARCHITECTURE "arm_aarch64")
        elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
            set(CPU_ARCHITECTURE "riscv64_riscv64")
        else()
            message(FATAL_ERROR "Unkown architecture ${CMAKE_SYSTEM_PROCESSOR}")
        endif()

        add_executable(${example} ${example_file_path})
        target_include_directories(${example} PRIVATE ${oneDAL_INCLUDE_DIRS})
        if (UNIX AND NOT APPLE)
            target_link_libraries(${example} PRIVATE -Wl,--start-group ${oneDAL_IMPORTED_TARGETS} -Wl,--end-group)
        else()
            target_link_libraries(${example} PRIVATE ${oneDAL_IMPORTED_TARGETS})
        endif()
        target_compile_options(${example} PRIVATE ${ONEDAL_CUSTOM_COMPILE_OPTIONS})
        target_link_options(${example} PRIVATE ${ONEDAL_CUSTOM_LINK_OPTIONS})
        if(WIN32 AND "${ONEDAL_LINK}" STREQUAL "dynamic" AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC|IntelLLVM")
            target_link_options(${example} PRIVATE /DEPENDENTLOADFLAG:0x2000)
        endif()
        set_target_properties(${example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/_cmake_results/${CPU_ARCHITECTURE}_${LINK_TYPE}")
    endforeach()
    set_common_compiler_options()
endfunction()

#Finction generate list of example paths based in user selection and exlude paths
function (generate_examples EXCLUDE_PATHS EXAMPLES_INPUT)
    if(EXAMPLES_INPUT)
        # Split the EXAMPLES_INPUT option into a list of patterns
        string(REPLACE "," ";" EXAMPLES_LIST ${EXAMPLES_INPUT})
        message(STATUS "Executing examples owerride: ${EXAMPLES_LIST}")
    else()
        set(EXAMPLES_LIST "")
    endif()

    # Initialize the EXAMPLES variable with an empty list
    set(EXAMPLES "")

    # Recursively find all the example files in the source directory, using the
    # EXCLUDE_PATHS exclude examples or directories
    file(GLOB_RECURSE EXAMPLE_FILES source/*/*.cpp)
    foreach(EXCLUDE_RULE ${EXCLUDE_PATHS})
        list(FILTER EXAMPLE_FILES EXCLUDE REGEX ${EXCLUDE_RULE})
    endforeach()

    # Convert the file names to executable names
    foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)
        if(EXAMPLES_LIST)
            # Otherwise, check if the example is in the EXAMPLES_LIST
            foreach(EXAMPLE_PATTERN ${EXAMPLES_LIST})
                if("${EXAMPLE_NAME}" MATCHES "${EXAMPLE_PATTERN}")
                    list(APPEND EXAMPLES ${EXAMPLE_FILE})
                endif()
            endforeach()
        else()
            # If EXAMPLES_LIST is empty, add all examples to the EXAMPLES variable
            list(APPEND EXAMPLES ${EXAMPLE_FILE})
        endif()
    endforeach()

    # Add the examples to the build
    add_examples("${EXAMPLES}")
endfunction()
