#! /bin/bash
#===============================================================================
# Copyright 2019 Intel Corporation
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

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ONEDAL_DIR=$(readlink -f "${SCRIPT_DIR}/../..")

function show_help_text {
    echo "Usage: $0"
    column -t -s":" <<< '--help:Display this information
--test-kind:Which tests to run. Must be one of [samples, examples]
--build-dir:The directory in which oneDAL was built
--compiler:The compiler suite to use to build the test programs
--interface:The interface to test, e.g. {oneapi,daal}/cpp
--conda-env:The Conda environment to load
--build-system:The type of build to perform, e.g. cmake
--backend:The backend C library to use. Must be one of [mkl, ref]
--platform:Explicitly pass the platform. This is the same as is passed to the top-level oneDAL build script
--cross-compile:Indicates whether cross-compilation is being performed
'
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --test-kind)
        TEST_KIND="$2"
        shift
        ;;
        --build-dir)
        BUILD_DIR="$2"
        shift
        ;;
        --compiler)
        compiler="$2"
        shift
        ;;
        --interface)
        interface="$2"
        shift
        ;;
        --conda-env)
        conda_env="$2"
        shift
        ;;
        --build-system)
        build_system="$2"
        shift
        ;;
        --backend)
        backend="$2"
        shift
        ;;
        --platform)
        platform="$2"
        shift
        ;;
        --cross-compile)
        cross_compile="yes"
        ;;
        --help)
        show_help_text
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

#Global exit code for testing script
TESTING_RETURN=0
PLATFORM=${platform:-$(bash dev/make/identify_os.sh)}
OS=${PLATFORM::3}
ARCH=${PLATFORM:3}
if [ "$ARCH" == "32e" ]; then
    full_arch=intel64
    arch_dir=intel_intel64
elif [ "$ARCH" == "arm" ]; then
    full_arch=arm
    arch_dir=arm_aarch64
elif [ "$ARCH" == "riscv64" ]; then
    full_arch=riscv64
    arch_dir=riscv64_riscv64
else
    echo "Unknown architecture ${ARCH} detected for platform ${PLATFORM}"
    exit 1
fi

build_system=${build_system:-cmake}
backend=${backend:-mkl}

if [ "${OS}" == "lnx" ]; then
    if [ -f /usr/share/miniconda/etc/profile.d/conda.sh ] ; then
        source /usr/share/miniconda/etc/profile.d/conda.sh
    fi
    if [ "${conda_env}" != "" ]; then
        conda activate "${conda_env}"
        echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
    fi
    compiler=${compiler:-gnu}
    link_modes=(static dynamic)
elif [ "${OS}" == "mac" ]; then
    if [ -f /usr/local/miniconda/etc/profile.d/conda.sh ] ; then
        source /usr/local/miniconda/etc/profile.d/conda.sh
    fi
    if [ "${conda_env}" != "" ]; then
        conda activate "${conda_env}"
        echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
    fi
    compiler=${compiler:-clang}
    link_modes=(static dynamic)
else
    echo "Error not supported OS: ${OS}"
    exit 1
fi

if [ "$(uname)" == "Linux" ]; then
    make_op="-j$(nproc --all)"
else
    make_op="-j$(sysctl -n hw.physicalcpu)"
fi

#setup env for DAL
source "${BUILD_DIR}"/daal/latest/env/vars.sh

#setup env for TBB
export TBBROOT="${TBBROOT:-${ONEDAL_DIR}/__deps/tbb/${OS}}"
export CPATH="${TBBROOT}/include${CPATH:+:$CPATH}"
export CMAKE_PREFIX_PATH="${TBBROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

if [ "${OS}" == "mac" ]; then
    export DYLD_LIBRARY_PATH=${TBBROOT}/lib:${DYLD_LIBRARY_PATH}
    export LIBRARY_PATH=${TBBROOT}/lib:${LIBRARY_PATH}
else
    if [ -d "${TBBROOT}/lib/${full_arch}/gcc4.8" ] ; then
        TBB_LIB_DIR="${TBBROOT}/lib/${full_arch}/gcc4.8"
    else
        TBB_LIB_DIR="${TBBROOT}/lib"
    fi
    export LD_LIBRARY_PATH=${TBB_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export LIBRARY_PATH=${TBB_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}
fi

interface=${interface:-daal/cpp}
pushd "${BUILD_DIR}/daal/latest/${TEST_KIND}/${interface}" || exit 1

for link_mode in "${link_modes[@]}"; do
    if [ "${link_mode}" == "static" ]; then
        lib_ext="a"
        l="lib"
    elif [ "${link_mode}" == "dynamic" ]; then
        lib_ext="so"
        if [ "${OS}" == "lnx" ]; then
            l="so"
        else
            l="dylib"
        fi
    fi
    if [ "$build_system" == "cmake" ]; then
        if [[ ${compiler} == gnu ]]; then
            CC=gcc
            CXX=g++
        elif [[ ${compiler} == clang ]]; then
            CC=clang
            CXX=clang++
        elif [[ ${compiler} == icx ]]; then
            CC=icx
            CXX=icpx
        fi

        export CC
        export CXX

        echo "============== Configuration: =============="
        echo Compiler:  "${compiler}"
        echo Link mode: "${link_mode}"
        echo CC: ${CC}
        echo CXX: ${CXX}
        echo "============================================"

        if [ -d "Build" ]; then
            rm -rf Build/*
        else
            mkdir Build
        fi

        if [ "${backend}" == "ref" ] ; then
            ref_backend="ON"
        else
            ref_backend="OFF"
        fi

        cmake_options=(-B Build
            -S .
            -G "Unix Makefiles"
            -DONEDAL_LINK="${link_mode}"
            -DREF_BACKEND="${ref_backend}")

        if [ "${cross_compile}" == "yes" ] ; then
            # Set the cmake toolchain file to set up the cross-compilation
            # correctly
            cmake_options+=(-DCMAKE_TOOLCHAIN_FILE="${ONEDAL_DIR}"/.ci/env/"${ARCH}"-"${compiler}"-crosscompile-toolchain.cmake)
        fi

        echo cmake "${cmake_options[@]}"
        cmake "${cmake_options[@]}"
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') CMAKE GENERATE FAILED\t\t"
            TESTING_RETURN=${err}
            continue
        fi
        make "${make_op}" -C Build
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') BUILD FAILED\t\t"
            TESTING_RETURN=${err}
            continue
        fi
        output_result=
        err=
        cmake_results_dir="_cmake_results/${arch_dir}_${lib_ext}"

        if [ "$TEST_KIND" = "samples" ]; then
            cd Build;
            cmake_results_dir="../_cmake_results/${arch_dir}_${lib_ext}"
        fi

        for p in "${cmake_results_dir}"/*; do
            e=$(basename "$p")

            if [ "$TEST_KIND" = "samples" ]; then
                run_command="make run_${e}"
            else
                run_command="$p"
            fi

            ${run_command} > "${e}".res 2>&1
            err=$?
            output_result=$(cat "${e}".res)
            mv -f "${e}".res ${cmake_results_dir}/
            status_ex=
            if [ ${err} -ne 0 ]; then
                echo "${output_result}"
                status_ex="$(date +'%H:%M:%S') FAILED\t\t${e} with errno ${err}"
                TESTING_RETURN=${err}
            else
                echo "${output_result}" | grep -i "error\|warn"
                status_ex="$(date +'%H:%M:%S') PASSED\t\t${e}"
            fi
            echo -e "$status_ex"
        done
        # Go back from "Build" directory in case of samples testing
        [ "$TEST_KIND" = "samples" ] && cd ..
    else
        build_command="make ${make_op} ${l}${full_arch} mode=build compiler=${compiler}"
        echo "Building ${TEST_KIND} ${build_command}"
        (${build_command})
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') BUILD FAILED\t\t${link_mode}"
            TESTING_RETURN=${err}
            continue
        else
            echo -e "$(date +'%H:%M:%S') BUILD COMPLETED\t\t${link_mode}"
        fi
        run_command="make ${l}${full_arch} mode=run compiler=${compiler}"
        echo "Running ${TEST_KIND} ${run_command}"
        (${run_command})
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') RUN FAILED\t\t${link_mode} with errno ${err}"
            TESTING_RETURN=${err}
            continue
        else
            echo -e "$(date +'%H:%M:%S') RUN PASSED\t\t${link_mode}"
        fi
    fi
done

popd || exit 1

#exit with overall testing status
exit ${TESTING_RETURN}
