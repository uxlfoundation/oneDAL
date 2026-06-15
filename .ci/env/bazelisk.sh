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
BAZELISK_BINARY=bazelisk-linux-amd64
BAZELISK_SHA256=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992

# Download Bazelisk
wget --tries=5 --waitretry=5 --retry-connrefused \
  "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/${BAZELISK_BINARY}"
echo "${BAZELISK_SHA256}  ${BAZELISK_BINARY}"
echo "${BAZELISK_SHA256}  ${BAZELISK_BINARY}" | sha256sum --check
# "Install" bazelisk
chmod +x "${BAZELISK_BINARY}"
mkdir -p bazel/bin
mv "${BAZELISK_BINARY}" bazel/bin/bazel
export BAZEL_VERSION=$(./bazel/bin/bazel --version | awk '{print $2}')
export PATH=$PATH:$(pwd)/bazel/bin
