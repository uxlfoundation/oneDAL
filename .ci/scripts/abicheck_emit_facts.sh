#!/bin/bash
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
#
# Compiles oneDAL's public umbrella headers through icpx with the
# abicheck-facts Clang plugin loaded, emitting source_facts/*.jsonl into an
# abicheck_inputs/ pack (the same protocol `abicheck dump --build-info`/
# `abicheck scan --build-info` ingest). This gives ISA-independent source-level
# coverage (macros, constexpr values, default arguments, inline/template
# bodies) of the full public API surface, regardless of which single ISA the
# accompanying binary build covers.
#
# Usage: abicheck_emit_facts.sh <plugin-so-path> <output-pack-dir>

set -eo pipefail

plugin_so=${1:?usage: abicheck_emit_facts.sh <plugin-so-path> <output-pack-dir>}
pack_dir=${2:?usage: abicheck_emit_facts.sh <plugin-so-path> <output-pack-dir>}

if [ -z "${CMPLR_ROOT}" ]; then
    echo "::error:: CMPLR_ROOT is not set. Source /opt/intel/oneapi/setvars.sh before calling this script."
    exit 1
fi

mkdir -p "${pack_dir}"

emit_facts() {
    local tu=$1
    local include_root=$2
    local public_roots=$3

    icpx -std=c++17 -fsycl -I"${include_root}" \
        -fplugin="${plugin_so}" \
        -Xclang -plugin-arg-abicheck-facts -Xclang "out=${pack_dir}" \
        -Xclang -plugin-arg-abicheck-facts -Xclang "public-roots=${public_roots}" \
        -fsyntax-only "${tu}"
}

# daal's public API surface
emit_facts cpp/daal/include/daal.h cpp/daal/include cpp/daal/include

# oneapi::dal's public API surface
emit_facts cpp/oneapi/dal.hpp cpp/oneapi cpp/oneapi/dal

echo "Emitted source facts to ${pack_dir}/source_facts/."
