#!/bin/bash
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

set -eo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ONEDAL_DIR=$(readlink -f "${SCRIPT_DIR}/../..")
TBB_DEFAULT_VERSION="v2021.10.0"

# Function to display help
show_help() {
    echo "Usage: $0 [--help]"
    column -t -s":" <<< '--help:Display this information
--CC <bin path>:Pass full path to c compiler. Default is GNU gcc.
--CXX <bin path>:Pass full path to c++ compiler. Default is GNU g++.
--cross-compile:Pass this flag if cross-compiling. This may override the CC and CXX variables
--toolchain-file <file>:Pass path to cmake toolchain file. To be used with --cross-compile
--target-arch <name>:Target architecture name for cross-compilation. Use only with '--cross-compile'. Must be one of [x86_64, aarch64]
--tbb-src <path>:The path to an existing TBB source directory. This is downloaded if not passed
--prefix <path>:The path to install oneTBB into. Defaults to ${ONEDAL_DIR}/__deps/tbb-$${target_arch}
--build-dir <path>:The path to build in. Defaults to ${ONEDAL_DIR}/__work/tbb-$${target_arch}
--version <version string>:The version of oneTBB to fetch and build. Must be a valid git reference in the upstream oneTBB repo
'
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --CXX)
        CXX="$2"
        shift;;
        --CC)
        CC="$2"
        shift;;
        --toolchain-file)
        toolchain_file="$2"
        shift;;
        --target-arch)
        target_arch="$2"
        shift;;
        --help)
        show_help
        exit 0
        ;;
        --cross-compile)
        cross_compile="yes"
        ;;
        --tbb-src)
        tbb_src="$2"
        shift;;
        --prefix)
        tbb_prefix="$2"
        shift;;
        --build-dir)
        build_dir="$2"
        shift;;
        --version)
        TBB_VERSION="$2"
        shift;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

set_arch_dir() {
    local arch="$1"
    arch_dir=""
    if [ "$arch" == "x86_64" ]; then
        arch_dir="intel64"
    elif [ "$arch" == "aarch64" ]; then
        arch_dir="arm"
    elif [ "$arch" == "riscv64" ]; then
        arch_dir="riscv64"
    else
        echo "Unsupported architecture '${arch}'. Quitting tbb build script."
        exit 1
    fi
}

if [ "${cross_compile}" == "yes" ]; then
    if [ -z "${toolchain_file}" ]; then
        echo "'toolchain_file' is not set, although we are performing cross-compilation"
        exit 1
    fi
    if [ ! -f "${toolchain_file}" ]; then
        echo "'${toolchain_file}' file does not exist. Please specify using '--toolchain-file <path>' argument."
        exit 1
    fi
    target_arch=${target_arch:-aarch64}
else
    target_arch=${target_arch:-$(uname -m)}
fi
if [[ -z "${arch_dir}" ]] ; then
    set_arch_dir "${target_arch}"
fi
build_dir=${build_dir:-${ONEDAL_DIR}/__work/tbb-$target_arch}
tbb_prefix=${tbb_prefix:-${ONEDAL_DIR}/__deps/tbb-$target_arch}

sudo apt-get update
sudo apt-get install build-essential gcc gfortran cmake -y
tbb_src=${tbb_src:-${ONEDAL_DIR}/__work/onetbb-src}
if [[ ! -d "${tbb_src}" ]] ; then
  TBB_VERSION="${TBB_VERSION:-${TBB_DEFAULT_VERSION}}"
  git clone --depth 1 --branch "${TBB_VERSION}" https://github.com/uxlfoundation/oneTBB.git "${tbb_src}"
fi

rm -rf "${tbb_prefix}"
mkdir -p "${build_dir}"
pushd "${build_dir}"
if [ "${cross_compile}" == "yes" ]; then
    cmake_options=(-DCMAKE_TOOLCHAIN_FILE="${toolchain_file}"
      -DCMAKE_BUILD_TYPE=Release
      -DTBB_TEST=OFF
      -DTBB_STRICT_PROTOTYPES=OFF
      -DCMAKE_INSTALL_PREFIX="${tbb_prefix}"
      "${tbb_src}"
    )

else
    # Set default values for CXX and CC
    CXX="${CXX:-g++}"
    CC="${CC:-gcc}"

    echo "CXX is set to: $CXX"
    echo "CC is set to: $CC"
    cmake_options=(-DCMAKE_CXX_COMPILER="${CXX}"
      -DCMAKE_C_COMPILER="${CC}"
      -DCMAKE_BUILD_TYPE=Release
      -DTBB_TEST=OFF
      -DTBB_STRICT_PROTOTYPES=OFF
      -DCMAKE_INSTALL_PREFIX="${tbb_prefix}"
      "${tbb_src}"
    )
fi

echo cmake "${cmake_options[@]}"
cmake "${cmake_options[@]}"
make -j"$(nproc --all)"
make install
popd

