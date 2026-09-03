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

BAZELISK_VERSION=v1.29.0

# Bazelisk itself always runs on the CI *exec* host (even when the build
# cross-compiles to another target arch, e.g. the riscv64 job below), so pick
# the asset matching the host running this script, not the oneDAL target arch.
case "$(uname -m)" in
  x86_64|amd64)
    BAZELISK_ASSET=bazelisk-linux-amd64
    ;;
  aarch64|arm64)
    BAZELISK_ASSET=bazelisk-linux-arm64
    ;;
  *)
    echo ":error: Unsupported host architecture for Bazelisk: $(uname -m)" >&2
    exit 1
    ;;
esac

# collect information about the bazelisk release
BAZELISK_JSON=$(wget -qO- \
  --header="Accept: application/vnd.github+json" \
  ${GITHUB_TOKEN:+--header="Authorization: Bearer $GITHUB_TOKEN"} \
  --header="X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/bazelbuild/bazelisk/releases/tags/$BAZELISK_VERSION)
if [ $? -ne 0 ] || [ -z "$BAZELISK_JSON" ]; then
  echo ":error: Failed to fetch Bazelisk release information from GitHub API." >&2
  exit 1
fi

# extract SHA256 from json
SHA256=""
found=""
while IFS= read -r line; do
  if [[ $line == *"\"name\": \"${BAZELISK_ASSET}\""* ]]; then
    found=1
  elif [[ $found && $line == *'"digest":'* ]]; then
    SHA256=$(echo "$line" | sed -n 's/.*"sha256:\([^"]*\)".*/\1/p')
    break
  fi
done < <(printf '%s\n' "$BAZELISK_JSON")
SHA256+="  ${BAZELISK_ASSET}"

# Download Bazelisk
wget https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/${BAZELISK_ASSET}
echo $SHA256
echo ${SHA256} | sha256sum --check
# "Install" bazelisk
chmod +x ${BAZELISK_ASSET}
mkdir -p bazel/bin
mv ${BAZELISK_ASSET} bazel/bin/bazel
export BAZEL_VERSION=$(./bazel/bin/bazel --version | awk '{print $2}')
export PATH=$PATH:$(pwd)/bazel/bin
