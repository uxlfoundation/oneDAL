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

BAZELISK_VERSION=v1.27.0
# collect information about the bazelisk release
BAZELISK_JSON=$(wget -qO- \
  --header="Accept: application/vnd.github+json" \
  ${GITHUB_TOKEN:+--header="Authorization: Bearer $GITHUB_TOKEN"} \
  --header="X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/bazelbuild/bazelisk/releases/tags/$BAZELISK_VERSION)

# extract SHA256 from json
SHA256=""
found=""
while IFS= read -r line; do
  if [[ $line == *'"name": "bazelisk-linux-amd64"'* ]]; then
    found=1
  elif [[ $found && $line == *'"digest":'* ]]; then
    SHA256=$(echo "$line" | sed -n 's/.*"sha256:\([^"]*\)".*/\1/p')
    break
  fi
done < <(printf '%s\n' "$BAZELISK_JSON")
SHA256+="  bazelisk-linux-amd64"

# Download Bazelisk
wget https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/bazelisk-linux-amd64
echo $SHA256
echo ${SHA256} | sha256sum --check
# "Install" bazelisk
chmod +x bazelisk-linux-amd64
mkdir -p bazel/bin
mv bazelisk-linux-amd64 bazel/bin/bazel
export BAZEL_VERSION=$(./bazel/bin/bazel --version | awk '{print $2}')
export PATH=$PATH:$(pwd)/bazel/bin
