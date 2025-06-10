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

include_guard()

# Defines mapping between link mode and filenames
function(set_link_type)
    if("${ONEDAL_LINK}" STREQUAL "static")
        set(LINK_TYPE "a" PARENT_SCOPE)
    else()
        set(LINK_TYPE "so" PARENT_SCOPE)
    endif()
endfunction()

# Define dependencies for MPI/CCL samples
function(find_dependencies)
    if(ONEDAL_DISTRIBUTED STREQUAL "yes")
        find_package(MPI REQUIRED)
        set(MPI_DEPENDENCIES MPI::MPI_C MPI::MPI_CXX PARENT_SCOPE)
        if(ONEDAL_USE_CCL STREQUAL "yes")
            # This policy allows finding modules using _ROOT variables
            cmake_policy(SET CMP0074 NEW)
            find_package(CCL REQUIRED)
            set(MPI_DEPENDENCIES MPI::MPI_C MPI::MPI_CXX ${CCL_LIBRARY} PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(set_common_compiler_options)
    # Setting base common set of params for samples compilation
    if(WIN32)
        add_compile_options(/W3 /EHsc)
    elseif(UNIX)
        add_compile_options(-pedantic -Wall -Wextra -Wno-unused-parameter)
    endif()

    if(ONEDAL_USE_CCL STREQUAL "no")
        add_compile_options(-Werror)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" AND WIN32 OR ONEDAL_INTERFACE STREQUAL "no")
        add_compile_options(-Wall -w)
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

# Function for adding new samples to CMAKE configuration based on list of samples paths
function(add_samples samples_paths)
    foreach(sample_file_path ${samples_paths})
        get_filename_component(sample ${sample_file_path} NAME_WE)

        # Detect CPU architecture
        if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
            set(CPU_ARCHITECTURE "intel_intel64")
        elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64")
            set(CPU_ARCHITECTURE "arm_aarch64")
        else()
            message(FATAL_ERROR "Unknown architecture ${CMAKE_HOST_SYSTEM_PROCESSOR}")
        endif()

        add_executable(${sample} ${sample_file_path})
        target_include_directories(${sample} PRIVATE ${oneDAL_INCLUDE_DIRS})

        if(ONEDAL_USE_CCL STREQUAL "yes")
            target_include_directories(${sample} PRIVATE ${CCL_INCLUDE_DIR})
        endif()

        if(UNIX AND NOT APPLE)
            target_link_libraries(${sample} PRIVATE -Wl,--start-group ${oneDAL_IMPORTED_TARGETS} ${MPI_DEPENDENCIES} -Wl,--end-group)
        else()
            target_link_libraries(${sample} PRIVATE ${oneDAL_IMPORTED_TARGETS} ${MPI_DEPENDENCIES})
        endif()

        target_compile_options(${sample} PRIVATE ${ONEDAL_CUSTOM_COMPILE_OPTIONS})
        target_link_options(${sample} PRIVATE ${ONEDAL_CUSTOM_LINK_OPTIONS})
        if(WIN32 AND "${ONEDAL_LINK}" STREQUAL "dynamic" AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC|IntelLLVM")
            target_link_options(${sample} PRIVATE /DEPENDENTLOADFLAG:0x2000)
        endif()
        set_target_properties(${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/_cmake_results/${CPU_ARCHITECTURE}_${LINK_TYPE}")

        add_custom_target(run_${sample}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                    ${MPIEXEC_MAX_NUMPROCS} -ppn ${MPIEXEC_NUMPROCS_PER_NODE} $<TARGET_FILE:${sample}>
            DEPENDS ${sample}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endforeach()
    set_common_compiler_options()
endfunction()

#Function generate list of sample paths based in user selection and exlude paths
function(generate_samples EXCLUDE_PATHS SAMPLES_INPUT)
    if(SAMPLES_INPUT)
        # Split the SAMPLES_INPUT option into a list of patterns
        string(REPLACE "," ";" SAMPLES_LIST ${SAMPLES_INPUT})
        message(STATUS "Executing samples override: ${SAMPLES_LIST}")
    else()
        set(SAMPLES_LIST "")
    endif()

    # Initialize the SAMPLES variable with an empty list
    set(SAMPLES "")

    # Recursively find all the sample files in the source directory, using the
    # EXCLUDE_PATHS exclude samples or directories
    file(GLOB_RECURSE SAMPLE_FILES sources/*.cpp)
    foreach(EXCLUDE_RULE ${EXCLUDE_PATHS})
        list(FILTER SAMPLE_FILES EXCLUDE REGEX ${EXCLUDE_RULE})
    endforeach()

    # Convert the file names to executable names
    foreach(SAMPLE_FILE ${SAMPLE_FILES})
        get_filename_component(SAMPLE_NAME ${SAMPLE_FILE} NAME_WE)
        if(SAMPLES_LIST)
            # Otherwise, check if the sample is in the SAMPLES_LIST
            foreach(SAMPLE_PATTERN ${SAMPLES_LIST})
                if("${SAMPLE_NAME}" MATCHES "${SAMPLE_PATTERN}")
                    list(APPEND SAMPLES ${SAMPLE_FILE})
                endif()
            endforeach()
        else()
            # If SAMPLES_LIST is empty, add all samples to the SAMPLES variable
            list(APPEND SAMPLES ${SAMPLE_FILE})
        endif()
    endforeach()

    # Add the samples to the build
    add_samples("${SAMPLES}")
endfunction()
