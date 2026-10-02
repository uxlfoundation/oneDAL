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

# Build image for the agent evals (repo-eval.yaml image.dockerfile): the toolchain the graders and the agent's
# CPU-only builds need, nothing else. No agent CLI (agent-benchmark brings the harness) and no oneAPI compilers
# (tasks use *_host targets). Build from the repository root:
#
#   docker build -f dev/agent_evals/Dockerfile -t onedal-agent-evals .
#
# Runs are offline (image.network: none). External Bazel repositories come from the shared repository cache, which
# is filled once with network access; see "Running under agent-benchmark" in README.md.

FROM ubuntu:24.04@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea

# GCC 13 is what the CPU-only builds of every task were validated with. unzip and zstd unpack the conda-forge
# MKL/TBB packages (dev/bazel/repos.bzl); without zstd the fetch fails silently and the build fails at analysis.
# Some task bases link -lOpenCL into *_host tests (ocl-icd-opencl-dev).
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

# bazelisk picks the Bazel version from each workspace's .bazelversion, and task bases pin different ones.
# Download them all now so no run fetches Bazel. Refresh the list with:
#   for b in $(jq -r .base dev/agent_evals/tasks/*/task.json | sort -u); do git show $b:.bazelversion; done | sort -u
ARG TASK_BAZEL_VERSIONS="6.2.0 7.0.2 8.4.1 9.0.1 9.1.0 9.1.1"
ENV BAZELISK_HOME=/opt/bazelisk
COPY .bazelversion /tmp/.bazelversion
RUN for v in $(cat /tmp/.bazelversion) ${TASK_BAZEL_VERSIONS}; do USE_BAZEL_VERSION=$v bazelisk --version || exit 1; done && \
    chmod -R a+rX /opt/bazelisk && rm /tmp/.bazelversion

# clang-format at the version CI pins (.pre-commit-config.yaml); the graders' format_ok uses it
RUN python3 -m venv /opt/clang-format && /opt/clang-format/bin/pip install --no-cache-dir clang-format==20.1.8
ENV ONEDAL_EVAL_CLANG_FORMAT=/opt/clang-format/bin/clang-format

# `bazel` is the eval wrapper: output base from EVAL_BAZEL_OUTPUT_BASE (per run), repository and disk caches shared
# across runs (repo-eval.yaml image.caches), MKLROOT always unset. Never set MKLROOT in this image: Bazel would
# re-fetch @mkl as symlinks into it.
COPY dev/agent_evals/bin/bazel /usr/local/bin/bazel
RUN chmod 755 /usr/local/bin/bazel
ENV ONEDAL_EVAL_BAZEL=/usr/local/bin/bazelisk \
    ONEDAL_EVAL_REPO_CACHE=/cache/bazel-repo \
    ONEDAL_EVAL_DISK_CACHE=/cache/bazel-disk \
    EVAL_BAZEL_REGISTRY=file:///cache/bazel-registry \
    HOME=/tmp/home
RUN mkdir -p /cache/bazel-repo /cache/bazel-disk /cache/bazel-registry /tmp/home && \
    chmod 1777 /cache/bazel-repo /cache/bazel-disk /cache/bazel-registry /tmp/home

# The Bazel server daemonizes; without an init as PID 1 nothing reaps it and `bazel shutdown` hangs then fails
ENTRYPOINT ["/usr/bin/tini", "--"]
