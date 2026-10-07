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

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# cquery writes loading/analysis chatter to stderr, so it is captured rather
# than inherited -- but only shown when the query itself fails. Swallowing it
# outright leaves `set -e` killing the script on a broken query with nothing in
# the log to say which query, or why.
query() {
    local expr="$1"
    shift
    if ! "${bazel_cmd}" cquery "${expr}" "${common[@]}" "$@" --output=label \
        >"${work}/query.out" 2>"${work}/query.err"; then
        echo "cquery failed: ${expr} ${*}" >&2
        cat "${work}/query.err" >&2
        return 1
    fi
    awk '{print $1}' "${work}/query.out" | sort -u
}

# Every assertion below says which pair it was checking when it fails. A bare
# `query ... | grep -q .` exits the script silently, which is indistinguishable
# in a CI log from the step dying for an unrelated reason.
assert_depends() {
    if ! query "somepath($1, $2)" --build_parameters_lib=no | grep -q .; then
        echo "expected ${1} to reach ${2} with --build_parameters_lib=no" >&2
        exit 1
    fi
}

assert_log_contains() {
    if ! grep -qi -- "$2" "$1"; then
        echo "expected ${3} to mention '${2}'; got:" >&2
        cat "$1" >&2
        exit 1
    fi
}

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
        assert_depends "${label}" "${module}"
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
        assert_depends "${label}" "${module}"
    done
done

# BUILD_PARAMETERS_LIB=no folds parameter code into the main libraries, so
# the release graph must not select any standalone parameter artifacts (above).
# Static parameter targets remain valid direct build targets, while the dynamic
# variants are intentionally incompatible in the folded layout.
for label in static_parameters static_parameters_dpc; do
    if ! "${bazel_cmd}" build "//cpp/oneapi/dal:${label}" "${common[@]}" --nobuild \
        --build_parameters_lib=no >"${work}/${label}.log" 2>&1; then
        echo "folded layout rejected direct target ${label}, which stays valid:" >&2
        cat "${work}/${label}.log" >&2
        exit 1
    fi
done
for label in dynamic_parameters dynamic_parameters_dpc; do
    if "${bazel_cmd}" build "//cpp/oneapi/dal:${label}" "${common[@]}" --nobuild \
        --build_parameters_lib=no >"${work}/${label}.log" 2>&1; then
        echo "folded layout unexpectedly permits direct target ${label}" >&2
        exit 1
    fi
    assert_log_contains "${work}/${label}.log" 'incompatible' "${label}'s rejection"
done

if "${bazel_cmd}" cquery @config//:validate_build_parameters_lib \
    --platforms=@config//:windows_analysis_platform --build_parameters_lib=yes \
    --noshow_progress >"${work}/windows.log" 2>&1; then
    echo "Windows --build_parameters_lib=yes unexpectedly passed validation" >&2
    exit 1
fi
assert_log_contains "${work}/windows.log" \
    '--build_parameters_lib=yes is not supported on Windows' \
    "the Windows validation failure"

echo "BUILD_PARAMETERS_LIB configured-graph checks passed"
