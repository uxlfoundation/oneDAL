#!/bin/bash
#===============================================================================
# Copyright 2021 Intel Corporation
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

component=$1

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/retry.sh"

function apt_install {
    retry sudo apt-get install -y "$@"
}

function update {
    retry sudo apt-get update
}

function fetch_intel_apt_key {
    (
        set -o pipefail
        wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
            gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
    )
}

function add_repo {
    retry fetch_intel_apt_key
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list
    update
}

function install_dpcpp {
    # DPC++ compiler version monitored by Renovate and sets exact value available via apt
    apt_install intel-oneapi-compiler-dpcpp-cpp=2026.2.1-990
}

function install_tbb {
    # TBB version monitored by Renovate and sets exact value available via apt
    apt_install intel-oneapi-tbb-devel=2023.2.0-492
}

function install_dpl {
    apt_install intel-oneapi-libdpstd-devel
}

function install_mkl {
    # MKL version monitored by Renovate and sets exact value available via apt
    apt_install intel-oneapi-mkl-devel=2026.2.2-7
    install_tbb
    install_dpl
}

function install_ccl {
    # The oneCCL samples also need the MPI compiler wrappers and mpiexec, which oneCCL does not pull in.
    apt_install intel-oneapi-ccl-devel intel-oneapi-mpi-devel
}

function install_clang-format {
    apt_install clang-format-20
}

function install_dev-base {
    apt_install gcc-multilib g++-multilib tree
}

function install_dev-base-conda {
    echo -e "default_channels:\n  - https://conda.anaconda.org/conda-forge" > ~/.condarc
    conda_retries
    conda env create -f .ci/env/environment.yml
}

function install_gnu-cross-compilers {
    apt_install "gcc-$1-linux-gnu" "g++-$1-linux-gnu" "gfortran-$1-linux-gnu"
}

function install_qemu_emulation_apt {
    retry sudo apt-get install -f -y
    apt_install qemu-user-static qemu-user binfmt-support
}

function install_opencl_apt {
    apt_install ocl-icd-opencl-dev
}

function install_qemu_emulation_deb {
    retry sudo apt-get install -f -y
    apt_install qemu-user qemu-user-binfmt qemu-user-static
    sudo systemctl restart systemd-binfmt.service
}

function install_llvm_version {
    apt_install curl
    retry curl -fsSL -o llvm.sh https://apt.llvm.org/llvm.sh
    chmod u+x llvm.sh
    retry sudo ./llvm.sh "$1"
    sudo update-alternatives --install /usr/bin/clang clang "/usr/bin/clang-$1" "${1}00"
    sudo update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/clang++-$1" "${1}00"
}

function build_sysroot {
    # Usage:
    #   build_sysroot working-dir arch os-name out-dir
    # where the architecture and OS name need to be recognised by debootstrap,
    # e.g. arch=arm64, os-name=jammy. The output directory path is relative to
    # the working directory
    mkdir -p "$1"
    pushd "$1" || exit
    apt_install debootstrap build-essential
    sudo debootstrap --arch="$2" --verbose --include=fakeroot,symlinks,libatomic1,libstdc++6 --resolve-deps --variant=minbase --components=main,universe "$3" "$4"
    sudo chroot "$4" symlinks -cr .
    sudo chown "${USER}" -R "$4"
    rm -rf "${4:?}"/{dev,proc,run,sys,var}
    rm -rf "${4:?}"/usr/{sbin,bin,share}
    rm -rf "${4:?}"/usr/lib/{apt,gcc,udev,systemd}
    rm -rf "${4:?}"/usr/libexec/gcc
    popd || exit
}

# Miniforge release monitored by Renovate; add the digests for every new version:
#   curl -L https://github.com/conda-forge/miniforge/releases/download/<version>/Miniforge3-<version>-Linux-<arch>.sh.sha256
MINIFORGE_VERSION=26.9.4-2

function conda_retries {
    # conda's own retries for repodata and package downloads (default 3, no backoff).
    conda config --set remote_max_retries 10
    conda config --set remote_backoff_factor 2
}

function fetch_miniforge {
    rm -f "${installer}"
    curl -fsSLO "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${installer}" && \
        echo "${sha256}  ${installer}" | sha256sum -c
}

function install_miniforge {
    local platform installer sha256

    platform="$(uname)-$(uname -m)"
    installer=Miniforge3-${MINIFORGE_VERSION}-${platform}.sh
    case "${MINIFORGE_VERSION}/${platform}" in
        26.9.4-2/Linux-x86_64)  sha256=bbaf16b3897da24c4be50315934ba74848f617b4495982841d6a0a60e0edbbdf ;;
        26.9.4-2/Linux-aarch64) sha256=1adf18b530e4ffb00886ce6d5bd2eb1fa6311d2f14709b13bd154d8cd98d94c8 ;;
        *)
            echo ":error: .ci/env/apt.sh has no pinned SHA256 for Miniforge ${MINIFORGE_VERSION} on ${platform}. Add it from:" >&2
            echo ":error:   curl -L https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${installer}.sha256" >&2
            exit 1
            ;;
    esac

    # A truncated download fails the digest check, so the check is inside the retried step.
    retry fetch_miniforge || exit $?

    sudo bash "${installer}" -b -p /usr/share/miniconda
    sudo chown -R "$(id -u):$(id -g)" /usr/share/miniconda
    source /usr/share/miniconda/etc/profile.d/conda.sh
}

function install_abigail {
    apt_install abigail-tools
}

if [ "${component}" == "dpcpp" ]; then
    add_repo
    install_dpcpp
elif [ "${component}" == "tbb" ]; then
    add_repo
    install_tbb
elif [ "${component}" == "mkl" ]; then
    add_repo
    install_mkl
elif [ "${component}" == "dpl" ]; then
    add_repo
    install_dpl
elif [ "${component}" == "ccl" ]; then
    add_repo
    install_ccl
elif [ "${component}" == "gnu-cross-compilers" ]; then
    update
    install_gnu-cross-compilers "$2"
elif [ "${component}" == "clang-format" ]; then
    update
    install_clang-format
elif [ "${component}" == "dev-base" ]; then
    update
    install_dev-base
elif [ "${component}" == "qemu-apt" ]; then
    update
    install_qemu_emulation_apt
elif [ "${component}" == "qemu-deb" ]; then
    update
    install_qemu_emulation_deb
elif [ "${component}" == "llvm-version" ] ; then
    update
    install_llvm_version "$2"
elif [ "${component}" == "build-sysroot" ] ; then
    update
    build_sysroot "$2" "$3" "$4" "$5"
elif [ "${component}" == "miniforge" ] ; then
    if [ ! -f /usr/share/miniconda/etc/profile.d/conda.sh ] ; then
        install_miniforge
    fi
    install_dev-base-conda
elif [ "${component}" == "abigail" ] ; then
    update
    install_abigail
elif [ "${component}" == "opencl" ]; then
    update
    install_opencl_apt
else
    echo "Usage:"
    echo "   $0 [dpcpp|tbb|mkl|dpl|ccl|gnu-cross-compilers|clang-format|dev-base|qemu-apt|qemu-deb|llvm-version|build-sysroot|miniforge|abigail|opencl]"
    exit 1
fi
