#!/bin/bash
#===============================================================================
# Copyright 2023 Intel Corporation
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

# Installs the pinned Bazelisk executable for CI. The PowerShell counterpart is
# `.ci/env/bazelisk.ps1`; keep the pinned version in step with it.
#
# The digests are pinned here rather than read from the GitHub release metadata,
# because the releases API is rate limited per IP and made `install-bazel` fail
# on unauthenticated runs. When bumping BAZELISK_VERSION, update every digest
# below as well; they are printed by
#
#   gh release view <version> -R bazelbuild/bazelisk \
#     --json assets --jq '.assets[] | "\(.name) \(.digest)"'
BAZELISK_VERSION=v1.29.0
BAZELISK_SHA256_amd64=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
BAZELISK_SHA256_arm64=e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010

# Bazelisk itself always runs on the CI *exec* host (even when the build
# cross-compiles to another target arch, e.g. the riscv64 job below), so pick
# the asset matching the host running this script, not the oneDAL target arch.
host_arch=$(uname -m)

case "${host_arch}" in
  x86_64|amd64)  arch=amd64; sha256=${BAZELISK_SHA256_amd64} ;;
  aarch64|arm64) arch=arm64; sha256=${BAZELISK_SHA256_arm64} ;;
  *)
    echo ":error: Unsupported host architecture for Bazelisk: ${host_arch}" >&2
    exit 1
    ;;
esac

BAZELISK_ASSET="bazelisk-linux-${arch}"

if ! wget -q "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/${BAZELISK_ASSET}"; then
  echo ":error: Failed to download ${BAZELISK_ASSET} ${BAZELISK_VERSION}." >&2
  exit 1
fi

# Checked explicitly: the script is sourced by most callers, so a bare
# `sha256sum --check` would only print FAILED and let the build go on with an
# unverified binary.
if ! echo "${sha256}  ${BAZELISK_ASSET}" | sha256sum --check --quiet; then
  echo ":error: SHA256 mismatch for ${BAZELISK_ASSET}. If you bumped" \
       "BAZELISK_VERSION, update the digests in $0." >&2
  rm -f "${BAZELISK_ASSET}"
  exit 1
fi

# "Install" bazelisk
chmod +x ${BAZELISK_ASSET}
mkdir -p bazel/bin
mv ${BAZELISK_ASSET} bazel/bin/bazel
export BAZEL_VERSION=$(./bazel/bin/bazel --version | awk '{print $2}')
export PATH=$PATH:$(pwd)/bazel/bin

# Callers that run this script instead of sourcing it lose the export above, so
# hand the directory to GitHub Actions the way `bazelisk.ps1` does.
if [ -n "${GITHUB_PATH}" ]; then
  echo "$(pwd)/bazel/bin" >> "${GITHUB_PATH}"
fi
