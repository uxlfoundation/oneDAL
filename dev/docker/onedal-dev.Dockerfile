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

FROM ubuntu:24.04@sha256:80dd3c3b9c6cecb9f1667e9290b3bc61b78c2678c02cbdae5f0fea92cc6734ab

ARG workdirectory="/sources/oneDAL"
WORKDIR ${workdirectory}

#Env setup
RUN apt-get update && \
      apt-get -y install sudo wget gnupg git make python3-setuptools doxygen software-properties-common

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3* -b -p /opt/conda

# Put conda in path to use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Installing environment for bazel
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64 && \
    chmod 755 bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/bin/bazel

COPY . ${workdirectory}

# Installing environment for base development dependencies
RUN .ci/env/apt.sh dev-base

# Installing environment for DPCPP development dependencies
RUN .ci/env/apt.sh dpcpp

# Installing environment for MKL development dependencies
RUN .ci/env/apt.sh mkl

# Installing environment for clang-format
RUN .ci/env/apt.sh clang-format

# Installing openBLAS dependency
RUN .ci/env/openblas.sh

# Installing oneTBB dependency
RUN ./dev/download_tbb.sh
