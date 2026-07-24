#!/usr/bin/env bash
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

# Cheap configured-graph validation for BUILD_PARAMETERS_LIB layouts.
set -euo pipefail

bazel_cmd="${BAZEL:-bazel}"
common=(--noshow_progress --cpu=avx2)
query() { "${bazel_cmd}" cquery "$1" "${common[@]}" "${@:2}" --output=label 2>/dev/null | awk '{print $1}' | sort -u; }

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

query 'labels(lib, //:release)' --build_parameters_lib=auto >"${work}/auto"
query 'labels(lib, //:release)' --build_parameters_lib=yes >"${work}/yes"
cmp "${work}/auto" "${work}/yes"

query 'labels(lib, //:release)' --build_parameters_lib=no >"${work}/no"
if grep -E ':(static|dynamic)_parameters(_dpc)?$' "${work}/no"; then
    echo "folded release still exposes parameter-library artifacts" >&2
    exit 1
fi
host_parameter_modules=(
    //cpp/oneapi/dal/algo:parameters
    //cpp/oneapi/dal/detail/parameters
)
dpc_parameter_modules=(
    //cpp/oneapi/dal/algo:parameters_dpc
    //cpp/oneapi/dal/detail/parameters:parameters_dpc
)
for label in //cpp/oneapi/dal:static //cpp/oneapi/dal:dynamic; do
    for module in "${host_parameter_modules[@]}"; do
        query "somepath(${label}, ${module})" --build_parameters_lib=no | grep -q .
    done
done
for label in //cpp/oneapi/dal:static_dpc //cpp/oneapi/dal:dynamic_dpc; do
    for module in "${host_parameter_modules[@]}"; do
        if query "somepath(${label}, ${module})" --build_parameters_lib=no | grep -q .; then
            echo "${label} depends on host parameter module ${module}" >&2
            exit 1
        fi
    done
    for module in "${dpc_parameter_modules[@]}"; do
        query "somepath(${label}, ${module})" --build_parameters_lib=no | grep -q .
    done
done

for label in static_parameters static_parameters_dpc dynamic_parameters dynamic_parameters_dpc; do
    if "${bazel_cmd}" build "//cpp/oneapi/dal:${label}" "${common[@]}" --nobuild \
        --build_parameters_lib=no >"${work}/${label}.log" 2>&1; then
        echo "folded layout unexpectedly permits direct target ${label}" >&2
        exit 1
    fi
    grep -qi 'incompatible' "${work}/${label}.log"
done

if "${bazel_cmd}" cquery @config//:validate_build_parameters_lib \
    --platforms=@config//:windows_analysis_platform --build_parameters_lib=yes \
    --noshow_progress >"${work}/windows.log" 2>&1; then
    echo "Windows --build_parameters_lib=yes unexpectedly passed validation" >&2
    exit 1
fi
grep -Fq -- '--build_parameters_lib=yes is not supported on Windows' "${work}/windows.log"

echo "BUILD_PARAMETERS_LIB configured-graph checks passed"
