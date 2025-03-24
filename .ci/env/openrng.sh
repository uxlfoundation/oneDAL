#!/bin/bash
#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
OPENRNG_DEFAULT_SOURCE_DIR="${ONEDAL_DIR}/__work/openrng"
OPENRNG_DEFAULT_VERSION="v24.04"

show_help() {
  echo "Usage: $0 [--help]"
  column -t -s":" <<< '--help:Display this information
--rng-src:The path to an existing OpenRNG source dircetory. The source is cloned if this parameter is omitted
--prefix:The path where OpenRNG will be installed
--CC <path>:Path to the C compiler executable to use
--CXX <path>:Path to the CXX compiler executable to use
--target-arch <name>:Target architecture name for cross-compilation. Use only with '--cross-compile'. Must be one of [x86_64, aarch64]
--cross-compile:Indicates that we are doing cross compilation
--version <version string>:The version of OpenRNG to fetch and build. Must be a valid git reference in the upstream OpenRNG repo
'
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --rng-src)
        rng_src_dir="$2"
        shift;;
        --prefix)
        PREFIX="$2"
        shift;;
        --CXX)
        CXX="$2"
        shift;;
        --CC)
        CC="$2"
        shift;;
        --LD)
        LD="$2"
        shift;;
        --target-arch)
        target="$2"
        shift;;
        --cross-compile)
        cross_compile="yes"
        ;;
        --version)
        OPENRNG_VERSION="$2"
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

CXX="${CXX:-g++}"
CC="${CC:-gcc}"
target="${target:-aarch64}"
OPENRNG_VERSION="${OPENRNG_VERSION:-${OPENRNG_DEFAULT_VERSION}}"
OPENRNG_DEFAULT_PREFIX="${ONEDAL_DIR}/__deps/openrng"
rng_prefix="${PREFIX:-${OPENRNG_DEFAULT_PREFIX}}"

rng_src_dir=${rng_src_dir:-$OPENRNG_DEFAULT_SOURCE_DIR}
if [[ ! -d "${rng_src_dir}" ]] ; then
    git clone --depth 1 --branch "${OPENRNG_VERSION}" https://git.gitlab.arm.com/libraries/openrng.git "${rng_src_dir}"
fi

if [ "${cross_compile}" == "yes" ]; then
    cmake_options=(-DCMAKE_INSTALL_PREFIX="${rng_prefix}"
        -DCMAKE_CXX_COMPILER="${CXX}"
        -DCMAKE_C_COMPILER="${CC}"
        -DCMAKE_SYSTEM_PROCESSOR="${target}"
        -DCMAKE_SYSTEM_NAME=linux
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_TESTING=OFF
    )
else
    cmake_options=(-DCMAKE_INSTALL_PREFIX="${rng_prefix}"
        -DCMAKE_CXX_COMPILER="${CXX}"
        -DCMAKE_C_COMPILER="${CC}"
        -DCMAKE_BUILD_TYPE=Release
    )
fi

pushd "${rng_src_dir}"
    rm -rf build
    mkdir build
        pushd build
            echo "${cmake_options[@]}"
            cmake "${cmake_options[@]}" ./../
            make install
        popd
popd
