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

BAZELISK_VERSION=v1.30.0

# Bazelisk itself always runs on the CI *exec* host (even when the build
# cross-compiles to another target arch, e.g. the riscv64 job below), so pick
# the asset matching the host running this script, not the oneDAL target arch.
host_arch=$(uname -m)

case "${host_arch}" in
  x86_64|amd64)  arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *)
    echo ":error: Unsupported host architecture for Bazelisk: ${host_arch}" >&2
    exit 1
    ;;
esac

BAZELISK_ASSET="bazelisk-linux-${arch}"

# Release asset digests, pinned so the install makes no rate-limited GitHub API
# call. Add the digests for every new BAZELISK_VERSION:
#   gh api repos/bazelbuild/bazelisk/releases/tags/<version> --jq '.assets[] | "\(.name) \(.digest)"'
case "${BAZELISK_VERSION}/${BAZELISK_ASSET}" in
  v1.30.0/bazelisk-linux-amd64) SHA256=a248ae2b2b62b5baebb748f6fba7203bff1a27e5d01b55399a8406744cb539e8 ;;
  v1.30.0/bazelisk-linux-arm64) SHA256=b2bcc040c213b44586b121bda6d983c35df121b917c3a7a66fb16f2fa0df1a4f ;;
  *)
    echo ":error: .ci/env/bazelisk.sh has no pinned SHA256 for ${BAZELISK_ASSET} at ${BAZELISK_VERSION}. Add it from:" >&2
    echo ":error:   gh api repos/bazelbuild/bazelisk/releases/tags/${BAZELISK_VERSION} --jq '.assets[] | \"\(.name) \(.digest)\"'" >&2
    exit 1
    ;;
esac
SHA256+="  ${BAZELISK_ASSET}"

# Release downloads fail transiently in CI (dropped connections, 5xx), so retry
# them with exponential backoff.
BAZELISK_FETCH_ATTEMPTS=${BAZELISK_FETCH_ATTEMPTS:-5}
BAZELISK_FETCH_DELAY=${BAZELISK_FETCH_DELAY:-5}
# A non-numeric attempt count would make the loop's test fail forever.
if ! [[ "${BAZELISK_FETCH_ATTEMPTS}" =~ ^[1-9][0-9]*$ && "${BAZELISK_FETCH_DELAY}" =~ ^[0-9]+$ ]]; then
  echo ":error: BAZELISK_FETCH_ATTEMPTS must be a positive integer and BAZELISK_FETCH_DELAY a non-negative integer." >&2
  exit 1
fi

# Download Bazelisk
attempt=1
delay=${BAZELISK_FETCH_DELAY}
until wget -O "${BAZELISK_ASSET}" \
  "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/${BAZELISK_ASSET}"; do
  if [ "${attempt}" -ge "${BAZELISK_FETCH_ATTEMPTS}" ]; then
    echo ":error: Failed to download ${BAZELISK_ASSET} after ${attempt} attempts." >&2
    exit 1
  fi
  echo "attempt ${attempt} of ${BAZELISK_FETCH_ATTEMPTS} failed, retrying in ${delay}s" >&2
  sleep "${delay}"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
done
unset attempt delay
echo $SHA256
# The script is sourced without `-e`, so a mismatch has to stop it explicitly.
if ! printf '%s\n' "${SHA256}" | sha256sum --check; then
  echo ":error: SHA256 mismatch for ${BAZELISK_ASSET}; refusing to install it." >&2
  rm -f "${BAZELISK_ASSET}"
  exit 1
fi
# "Install" bazelisk
chmod +x ${BAZELISK_ASSET}
mkdir -p bazel/bin
mv ${BAZELISK_ASSET} bazel/bin/bazel
export BAZEL_VERSION=$(./bazel/bin/bazel --version | awk '{print $2}')
export PATH=$PATH:$(pwd)/bazel/bin
