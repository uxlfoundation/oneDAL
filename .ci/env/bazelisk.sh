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

# Download Bazelisk
VERSION=v1.18.0
export SHA256="ce52caa51ef9e509fb6b7e5ad892e5cf10feb0794b0aed4d2f36adb00a1a2779  bazelisk-linux-amd64"
wget https://github.com/bazelbuild/bazelisk/releases/download/$VERSION/bazelisk-linux-amd64
echo ${SHA256} | sha256sum --check
# "Install" bazelisk
chmod +x bazelisk-linux-amd64
mkdir -p bazel/bin
mv bazelisk-linux-amd64 bazel/bin/bazel
export BAZEL_VERSION=$(./bazel/bin/bazel --version | awk '{print $2}')
export PATH=$PATH:$(pwd)/bazel/bin
