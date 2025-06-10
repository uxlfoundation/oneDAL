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

os=$(uname)
ARCH=$(uname -m)
if [ "${os}" = "Linux" ]; then
  if [ "${ARCH}" = "x86_64" ]; then
    echo lnx32e
  elif [ "${ARCH}" = "aarch64" ]; then
    echo lnxarm
  else
    echo "Unknown architecture: ${ARCH}"
    exit 1
  fi
elif [ "${os}" = "Darwin" ]; then
  echo mac32e
elif [[ "${os}" =~ "MSYS" || "${os}" =~ "CYGWIN" ]]; then
  echo win32e
else
  echo "Unknown OS: ${os}"
fi
