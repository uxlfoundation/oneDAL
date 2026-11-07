#===============================================================================
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

# Eval image: CPU-only, no agent CLI. Tasks are never COPYed: the agent must not read them.
# Runs are offline: Bazel repos come from the shared cache, filled once online (README, Eval image).

FROM ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea

# GCC 13 validated every CPU-only task build; zstd and unzip unpack the conda MKL/TBB packages.
# Some task bases link -lOpenCL into *_host tests, hence ocl-icd-opencl-dev.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git ocl-icd-opencl-dev patch python3 python3-venv tini unzip zip zstd && \
    rm -rf /var/lib/apt/lists/*

ARG BAZELISK_VERSION=v1.29.0
ARG BAZELISK_SHA256=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
RUN curl -fsSL -o /usr/local/bin/bazelisk \
      "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" && \
    echo "${BAZELISK_SHA256}  /usr/local/bin/bazelisk" | sha256sum -c - && \
    chmod 755 /usr/local/bin/bazelisk

# Holds every task base's Bazel version and the eval tree's own pin: CI builds the image from merged main.
# Keep it in step when a task base or the pinned .bazelversion changes.
ARG TASK_BAZEL_VERSIONS="6.2.0 7.0.2 8.4.1 9.0.1 9.1.0 9.1.1 9.2.0"
ENV BAZELISK_HOME=/opt/bazelisk
COPY .bazelversion /tmp/.bazelversion
RUN for v in $(cat /tmp/.bazelversion) ${TASK_BAZEL_VERSIONS}; do USE_BAZEL_VERSION=$v bazelisk --version || exit 1; done && \
    chmod -R a+rX /opt/bazelisk && rm /tmp/.bazelversion

# clang-format at the version CI pins (.pre-commit-config.yaml); the graders' format_ok uses it
RUN python3 -m venv /opt/clang-format && /opt/clang-format/bin/pip install --no-cache-dir clang-format==20.1.8
ENV ONEDAL_EVAL_CLANG_FORMAT=/opt/clang-format/bin/clang-format

# build_make_gnu: oneMKL and oneTBB for the make build, which the task prompt says are here; the task has no
# network. Pins as in graders/make.py MAKE_DEPS. Not on any default path, and MKLROOT stays unset (see below).
RUN python3 -m venv /opt/onedal-make-deps && \
    /opt/onedal-make-deps/bin/pip install --no-cache-dir mkl-static==2026.1.0 mkl-include==2026.1.0 tbb-devel==2023.1.0
ENV ONEDAL_EVAL_MAKE_DEPS=/opt/onedal-make-deps

# `bazel` is the eval wrapper: per-run output base, shared caches, MKLROOT always unset.
# Never set MKLROOT here: Bazel would re-fetch @mkl as symlinks into it.
COPY dev/agent_evals/bin/bazel /usr/local/bin/bazel
RUN chmod 755 /usr/local/bin/bazel
ENV ONEDAL_EVAL_BAZEL=/usr/local/bin/bazelisk \
    ONEDAL_EVAL_REPO_CACHE=/cache/bazel-repo \
    ONEDAL_EVAL_DISK_CACHE=/cache/bazel-disk \
    EVAL_BAZEL_REGISTRY=file:///cache/bazel-registry \
    ONEDAL_EVAL_INSTALL_ROOT=/cache/bazel-install \
    HOME=/tmp/home
RUN mkdir -p /cache/bazel-repo /cache/bazel-disk /cache/bazel-registry /cache/bazel-install /tmp/home && \
    chmod 1777 /cache/bazel-repo /cache/bazel-disk /cache/bazel-registry /cache/bazel-install /tmp/home

# The Bazel server daemonizes; without an init as PID 1 nothing reaps it and `bazel shutdown` hangs then fails.
# -s: under `docker run --init` (as dokimi runs every container) tini is not PID 1 and reaps as a subreaper
ENTRYPOINT ["/usr/bin/tini", "-s", "--"]
