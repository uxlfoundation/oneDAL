#!/bin/bash
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

set -eo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ONEDAL_DIR=$(readlink -f "${SCRIPT_DIR}/../..")
OPENBLAS_DEFAULT_SOURCE_DIR="${ONEDAL_DIR}/__work/openblas"
BLAS_DEFAULT_VERSION="v0.3.28"

show_help() {
  echo "Usage: $0 [--help]"
  column -t -s":" <<< '--help:Display this information
--target <target>:The OpenBLAS target to build for
--target-arch <arch>:The target architecture to build for. This is used when generating install directory names, and should match the target architecture for building oneDAL
--compiler <path>:Path to the C compiler executable to use
--host-compiler <path>:Path to the host compiler (suggests that that the target compiler given with '--compiler' is for another platform)
--cflags <flags>:Any extra flags to be added to the C compile line
--cross-compile:Indicates that we are doing cross compilation
--blas-src:The path to an existing OpenBLAS source dircetory. The source is cloned if this parameter is omitted
--prefix:The path where OpenBLAS will be installed
--version:The version of OpenBLAS to install. This is a git reference from the OpenBLAS repo, and defaults to ${BLAS_DEFAULT_VERSION}
--sysroot:If cross-compiling with LLVM, determines the location of the target architecture sysroot
--ilp64 <on/off>: whether or not to use the ILP64 build
'
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --target)
        target="$2"
        shift;;
        --target-arch)
        target_arch="$2"
        shift;;
        --compiler)
        compiler="$2"
        shift;;
        --host-compiler)
        host_compiler="$2"
        shift;;
        --cflags)
        cflags="$2"
        shift;;
        --cross-compile)
        cross_compile="yes"
        ;;
        --blas-src)
        blas_src_dir="$2"
        shift;;
        --prefix)
        PREFIX="$2"
        shift;;
        --version)
        BLAS_VERSION="$2"
        shift;;
        --sysroot)
        sysroot="$2"
        shift;;
        --ilp64)
        ilp64=on
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

target=${target:-ARMV8}
host_compiler=${host_compiler:-gcc}
compiler=${compiler:-aarch64-linux-gnu-gcc}
openblas_ilp64=${ilp64:-on}

target_arch=${target_arch:-$(uname -m)}
OPENBLAS_DEFAULT_PREFIX="${ONEDAL_DIR}/__deps/openblas_${target_arch}"
blas_prefix="${PREFIX:-${OPENBLAS_DEFAULT_PREFIX}}"
BLAS_VERSION="${BLAS_VERSION:-${BLAS_DEFAULT_VERSION}}"

if [ "${target_arch}" == "arm" ] ; then
  ARCH=aarch64
elif [ "${target_arch}" == riscv64 ] ; then
  ARCH=riscv64
fi

if [[ ${compiler} =~ "clang" && "${cross_compile}" == "yes" ]] ; then
  if [[ -z "${sysroot}" ]] ; then
    echo "Must supply --sysroot option if cross-compiling with a clang compiler"
    exit 1
  fi
fi

sudo apt-get update
sudo apt-get -y install build-essential gcc gfortran
blas_src_dir=${blas_src_dir:-$OPENBLAS_DEFAULT_SOURCE_DIR}
if [[ ! -d "${blas_src_dir}" ]] ; then
  git clone --depth=1 --branch "${BLAS_VERSION}" https://github.com/OpenMathLib/OpenBLAS "${blas_src_dir}"
fi

CoreCount=$(nproc --all)
pushd "${blas_src_dir}"
  # oneDAL does not need the Fortan interface. To avoid carrying around a false
  # dependence on libgfortran, we set NO_FORTRAN=1
  # Multi-threading is done through oneTBB, so we don't want OpenBLAS to spawn
  # threads through either OpenMP or pthreads. We set USE_OPENMP=0,
  # USE_THREAD=0.
  # The library may still be used in a multithreaded environment, so we set
  # USE_LOCKING=1 to ensure thread safety
  if [ "${cross_compile}" == "yes" ]; then
    if [[ ${compiler} =~ "clang" ]] ; then
      # Cross-compilation for clang needs to set a few extra variables, and we
      # need to make sure that we have a sysroot available. The sysroot is set
      # up outside of this script
      make_options=(-j"${CoreCount}"
        TARGET="${target}"
        HOSTCC="${host_compiler}"
        CC="${compiler}"
        LD="${compiler}"
        CXX="${compiler/clang/clang++}"
        AS="${compiler/clang/clang++}"
        NO_FORTRAN=1
        USE_OPENMP=0
        USE_THREAD=0
        USE_LOCKING=1
        ARCH="${ARCH}"
        CFLAGS="--target=${ARCH}-linux-gnu --sysroot ${sysroot} ${cflags}")
    else
      make_options=(-j"${CoreCount}"
          TARGET="${target}"
          HOSTCC="${host_compiler}"
          CC="${compiler}"
          NO_FORTRAN=1
          USE_OPENMP=0
          USE_THREAD=0
          USE_LOCKING=1
          CFLAGS="${cflags}")
    fi
  else
    make_options=(-j"${CoreCount}"
        NO_FORTRAN=1
        USE_OPENMP=0
        USE_THREAD=0
        USE_LOCKING=1)
  fi
  if [ "${openblas_ilp64}" == "on" ]; then
      make_options+=( 'BINARY=64' 'INTERFACE64=1' )
  fi
  # Clean
  echo make "${make_options[@]}" clean
  make "${make_options[@]}" clean
  # Build
  echo make "${make_options[@]}"
  make "${make_options[@]}"
  # The install needs to be done with the same options as the build
  make install "${make_options[@]}" PREFIX="${blas_prefix}"
popd
