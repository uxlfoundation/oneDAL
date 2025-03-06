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

set -eo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ONEDAL_DIR=$(readlink -f "${SCRIPT_DIR}/../../")

show_help() {
    echo "Usage: $0 [--help]"
    column -t -s":" <<< '--help:Show this help message
--compiler:The compiler toolchain to use. This is a value that is recognised by the oneDAL top level Makefile, and must be one of [gnu, clang, icx]
--optimizations:The microarchitecture to optimize the build for. This is a value that is recognised by the oneDAL top level Makefile
--target:The oneDAL target to build. This is passed directly to the oneDAL top level Makefile. Multiple targets can be passed by supplying a space-separated string as an argument
--backend-config:The optimised backend CPU library to use. Must be one of [mkl, ref]
--conda-env:The name of the conda environment to load
--cross-compile:Indicates that the target platform to build for is not the host platform
--plat:The platform to build for. This is passed to the oneDAL top level Makefile
--blas-dir:The BLAS installation directory to use to build oneDAL with in the case that the backend is given as `ref`. If the installation directory does not exist, attempts to build this from source
--tbb-dir:The TBB installation directory to use to build oneDAL with in the case that the backend is given as `ref`. If the installation directory does not exist, attempts to build this from source
--use-openrng:Set this to yes if openrng is to be used as RNG backend. Use this only with the `ref` backend.
--openrng-dir:The directory in which openrng is installed. Only to be used with `ref` backend and `--use-openrng yes`.
--sysroot:The sysroot to use, in the case that clang is used as the cross-compiler
'
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --compiler)
        compiler="$2"
        shift;;
        --optimizations)
        optimizations="$2"
        shift;;
        --target)
        target="$2"
        shift;;
        --backend-config)
        backend_config="$2"
        shift;;
        --conda-env)
        conda_env="$2"
        shift;;
        --cross-compile)
        cross_compile="yes"
        ;;
        --plat)
        PLAT="$2"
        shift;;
        --blas-dir)
        BLAS_INSTALL_DIR=$(readlink -f "$2")
        shift;;
        --openrng-dir)
        OPENRNG_INSTALL_DIR=$(readlink -f "$2")
        shift;;
        --tbb-dir)
        TBB_INSTALL_DIR=$(readlink -f "$2")
        shift;;
        --sysroot)
        sysroot="$2"
        shift;;
        --use-openrng)
        use_openrng="$2"
        shift;;
        --help)
        show_help
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

PLAT=${PLAT:-$(bash "${ONEDAL_DIR}"/dev/make/identify_os.sh)}
OS=${PLAT::3}
ARCH=${PLAT:3}

backend_config=${backend_config:-mkl}

if [ "${OS}" == "lnx" ]; then
    if [ "${conda_env}" != "" ]; then
        conda_init_path=/usr/share/miniconda/etc/profile.d/conda.sh
        if [ -f ${conda_init_path} ] ; then
            source ${conda_init_path}
            conda activate ${conda_env}
            echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
        fi
    fi
    compiler=${compiler:-gnu}

    #gpu support is only for Linux 64 bit
    if [ "${ARCH}" == "32e" ]; then
            with_gpu="true"
    else
            with_gpu="false"
    fi
elif [ "${OS}" == "mac" ]; then
    if [ "${conda_env}" != "" ]; then
        conda_init_path=/usr/local/miniconda/etc/profile.d/conda.sh
        if [ -f ${conda_init_path} ]; then
            source ${conda_init_path}
            conda activate ${conda_env}
            echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
        fi
    fi
    compiler=${compiler:-clang}
    with_gpu="false"
else
    echo "Error not supported OS: ${OS}"
    exit 1
fi

#setting build parallelization based on number of threads
if [ "$(uname)" == "Linux" ]; then
    make_op="-j$(nproc --all)"
else
    make_op="-j$(sysctl -n hw.physicalcpu)"
fi

# Override the compilers. We know which compilers we want in
# the case that we are using a GNU or LLVM toolchain
if [ "${ARCH}" == "arm" ] && [ "${cross_compile}" == "yes" ] && [ "${compiler}" == "gnu" ] ; then
    export CXX=aarch64-linux-gnu-g++
    export CC=aarch64-linux-gnu-gcc
elif [ "${compiler}" == "clang" ] ; then
    export CXX=clang++
    export CC=clang
elif [ "${compiler}" == "gnu" ] ; then
    export CXX=g++
    export CC=gcc
elif [ "${compiler}" == "icx" ] ; then
    export CXX=icpx
    export CC=icx
else
    echo "Unsupported compiler '${compiler}'"
    exit 1
fi

if [ "${cross-compile}" == "yes" ] && [ "${compiler}" == "clang" ] ; then
    if [[ -z "${sysroot}" ]] ; then
        echo "--sysroot must be specified when cross-compiling with clang"
        exit 1
    fi
    export ONEDAL_SYSROOT="${sysroot}"
fi

#main actions
echo "Call env scripts"
if [ "${backend_config}" == "mkl" ]; then
    source /opt/intel/oneapi/mkl/latest/env/vars.sh
elif [ "${backend_config}" == "ref" ] && [ ! -z "${BLAS_INSTALL_DIR}" ]; then
    export OPENBLASROOT="${BLAS_INSTALL_DIR}"
elif [ "${backend_config}" == "ref" ]; then
    echo "Sourcing ref(openblas) env"
    if [ ! -d "${ONEDAL_DIR}/__deps/openblas_${ARCH}" ]; then
        openblas_options=(--target-arch "${ARCH}")
        if [ "${cross_compile}" == "yes" ] ; then
            openblas_options+=(--host-compiler gcc
                --compiler "${CC}"
                --cross-compile)
            if [ "${optimizations}" == "sve" ] ; then
                openblas_options+=(--target ARMV8
                    --cflags -march=armv8-a+sve)
            elif [ "${optimizations}" == "rv64" ] ; then
                openblas_options+=(--target RISCV64_ZVL128B)
            fi

            if [ "${compiler}" == "clang" ] ; then
                openblas_options+=(--sysroot "${sysroot}")
            fi
        fi
        echo "${ONEDAL_DIR}"/.ci/env/openblas.sh "${openblas_options[@]}"
        "${ONEDAL_DIR}"/.ci/env/openblas.sh "${openblas_options[@]}"
    fi
    export OPENBLASROOT="${ONEDAL_DIR}/__deps/openblas_${ARCH}"
else
    echo "Not supported backend env"
fi

echo "Sourcing ref(openrng) env"
if [ "${backend_config}" == "ref" ] && [ "${use_openrng}" == "yes" ]; then
    if [ ! -z "${OPENRNG_INSTALL_DIR}" ]; then
        export OPENRNGROOT="${OPENRNG_INSTALL_DIR}"
    elif [ "${ARCH}" != "arm" ]; then
        echo "OpenRNG backend is currently only supported on ARM"
        exit 1
    elif [ ! -d "${ONEDAL_DIR}"/__deps/openrng ]; then
        openrng_options=(--target-arch aarch64)
        if [ "${cross_compile}" == "yes" ] ; then
            openrng_options+=(--cross-compile --CC aarch64-linux-gnu-gcc --CXX aarch64-linux-gnu-g++)
        fi
        echo "${ONEDAL_DIR}"/.ci/env/openrng.sh "${openrng_options[@]}"
        "${ONEDAL_DIR}"/.ci/env/openrng.sh "${openrng_options[@]}"
        export OPENRNGROOT="${ONEDAL_DIR}/__deps/openrng"
    else
        export OPENRNGROOT="${ONEDAL_DIR}/__deps/openrng"
    fi
fi

# TBB setup
if [[ ! -z "${TBB_INSTALL_DIR}" ]] ; then
    export TBBROOT="${TBB_INSTALL_DIR}"
    export LD_LIBRARY_PATH="${TBBROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [ "${ARCH}" == "32e" ] && [ "${backend_config}" == "ref" ]; then
    "${ONEDAL_DIR}"/dev/download_tbb.sh
elif [[ "${ARCH}" == "arm" || ("${ARCH}" == "riscv64") ]]; then
    if [[ "${ARCH}" == "arm" ]] ; then
        ARCH_STR=aarch64
    else
        # RISCV64
        ARCH_STR="${ARCH}"
    fi

    if [[ "${cross_compile}" == "yes" ]]; then
        tbb_options=(--cross-compile
          --toolchain-file
          "${ONEDAL_DIR}"/.ci/env/${ARCH}-${compiler}-crosscompile-toolchain.cmake
          --target-arch ${ARCH_STR}
        )
        echo "${ONEDAL_DIR}"/.ci/env/tbb.sh "${tbb_options[@]}"
        "${ONEDAL_DIR}"/.ci/env/tbb.sh "${tbb_options[@]}"
    else
        "${ONEDAL_DIR}"/.ci/env/tbb.sh
    fi
    export TBBROOT="$ONEDAL_DIR/__deps/tbb-${ARCH_STR}"
    export LD_LIBRARY_PATH=${TBBROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

make_options=("${target:-onedal_c}"
    "${make_op}"
    COMPILER="${compiler}"
    REQCPU="${optimizations}"
    BACKEND_CONFIG="${backend_config}"
    PLAT="${PLAT}"
)

if [ "${cross_compile}" == "yes" ] && [ "${compiler}" == "clang" ] ; then
    make_options+=(SYSROOT="${sysroot}")
fi

if [ "${use_openrng}" == "yes" ]; then
    make_options+=(RNG_BACKEND=openrng)
fi

echo "Calling make"
echo "CXX=$CXX"
echo "CC=$CC"
echo make "${make_options[@]}"
make "${make_options[@]}"
