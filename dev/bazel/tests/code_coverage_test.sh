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

# Positive-path checks for --code_coverage=true on the icx toolchain.
#
#   code_coverage_test.sh [--dpc] [extra bazel args...]
#
# Default (host) mode builds DAAL core and confirms the compile actions carry
# the Make-equivalent coverage flags while the separately built threading module
# does not. `--dpc` checks the option that only the DPC++ link adds,
# `-Xscoverage`; it is an on-demand check because it needs the full oneAPI
# runtime, while the host mode runs in public PR CI.
set -euo pipefail

mode="host"
if [[ "${1:-}" == "--dpc" ]]; then
    mode="dpc"
    shift
fi

bazel_cmd="${BAZEL:-bazel}"
startup_args=()
if [[ -n "${BAZEL_OUTPUT_USER_ROOT:-}" ]]; then
    startup_args+=("--output_user_root=${BAZEL_OUTPUT_USER_ROOT}")
fi
common_args=(--code_coverage=true)
if [[ "${mode}" == "dpc" ]]; then
    common_args+=(--release_dpc=true)
else
    # `-coverage` makes the compiler write a `.gcno` notes file next to the
    # object file, and that file is not a declared output of the compile action,
    # so a sandboxed action discards it on teardown and no gcov/lcov report can
    # be built from the run. Local execution keeps it in the output tree; this is
    # asserted below and documented in dev/bazel/README.md. The DPC++ mode makes
    # no such assertion, so it does not need local execution.
    common_args+=(--spawn_strategy=local)
fi
common_args+=("$@")
if [[ -n "${BAZEL_DISK_CACHE:-}" ]]; then
    common_args+=("--disk_cache=${BAZEL_DISK_CACHE}")
fi
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

if [[ "${mode}" == "host" ]]; then
    "${bazel_cmd}" "${startup_args[@]}" build //cpp/daal:core_static "${common_args[@]}"

    # The instrumentation is only usable if the notes files survive the build.
    output_path="$("${bazel_cmd}" "${startup_args[@]}" info output_path "${common_args[@]}")"
    if [[ -z "$(find -L "${output_path}" -name '*.gcno' -print -quit)" ]]; then
        echo "ERROR: no .gcno notes files under ${output_path} after a coverage build" >&2
        exit 1
    fi
    echo "coverage notes files (.gcno) are present in the output tree"

    "${bazel_cmd}" "${startup_args[@]}" aquery 'mnemonic("CppCompile", deps(//cpp/daal:core_static))' \
        "${common_args[@]}" --output=jsonproto >"${work}/core_actions.json"
    python3 -c "
import json, sys
with open('${work}/core_actions.json') as f:
    data = json.load(f)
target_actions = [a for a in data.get('actions', [])
                   if any('error_handling.cpp' in arg for arg in a.get('arguments', []))]
if not target_actions:
    sys.exit('ERROR: no compile action found for error_handling.cpp')
for action in target_actions:
    args = action.get('arguments', [])
    if '-coverage' not in args:
        sys.exit('ERROR: -coverage missing from compile arguments: {}'.format(args))
    if '-DGCOV_BUILD' not in args:
        sys.exit('ERROR: -DGCOV_BUILD missing from compile arguments: {}'.format(args))
print('error_handling.cpp compile action carries -coverage and -DGCOV_BUILD')
"

    "${bazel_cmd}" "${startup_args[@]}" build //cpp/daal:thread_static "${common_args[@]}"
    "${bazel_cmd}" "${startup_args[@]}" aquery 'mnemonic("CppCompile", deps(//cpp/daal:thread_static))' \
        "${common_args[@]}" --output=jsonproto >"${work}/thread_actions.json"
    python3 -c "
import json, sys
with open('${work}/thread_actions.json') as f:
    data = json.load(f)
actions = data.get('actions', [])
if not actions:
    sys.exit('ERROR: no compile action found for //cpp/daal:thread_static')
for action in actions:
    if '-coverage' not in action.get('arguments', []):
        sys.exit('ERROR: -coverage missing from threading_tbb compile arguments')
    if '-DGCOV_BUILD' in action.get('arguments', []):
        sys.exit('ERROR: threading_tbb compile action unexpectedly defines GCOV_BUILD')
print('threading_tbb compile actions carry -coverage and do not define GCOV_BUILD')
"

    echo "code-coverage icx smoke checks passed"
    exit 0
fi

target="//cpp/oneapi/dal/table:table_dpc"
"${bazel_cmd}" "${startup_args[@]}" build "${target}" "${common_args[@]}"
"${bazel_cmd}" "${startup_args[@]}" aquery "deps(${target})" \
    "${common_args[@]}" --output=jsonproto >"${work}/actions.json"

python3 - "${work}/actions.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    actions = json.load(f).get("actions", [])

dpc_links = [a for a in actions
             if a.get("mnemonic") in ("CppLink", "CppLinkDynamicLibrary")
             and any("table_dpc" in arg for arg in a.get("arguments", []))]
if not dpc_links:
    sys.exit("ERROR: no DPC++ link action found for table_dpc")
for action in dpc_links:
    if "-Xscoverage" not in action.get("arguments", []):
        sys.exit("ERROR: -Xscoverage missing from DPC++ link: {}".format(action.get("arguments", [])))
print("table_dpc link action carries -Xscoverage")
PY

# The released library is the link that matters most, but building all of oneAPI
# DAL with DPC++ is far more than a smoke needs. `aquery` answers the same
# question from analysis alone, without executing the link.
released="//cpp/oneapi/dal:dynamic_dpc"
"${bazel_cmd}" "${startup_args[@]}" aquery "${released}" \
    "${common_args[@]}" --output=jsonproto >"${work}/released_actions.json"

python3 - "${work}/released_actions.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    actions = json.load(f).get("actions", [])

links = [a for a in actions
         if a.get("mnemonic", "").startswith("CppLink")
         and any("libonedal_dpc" in arg for arg in a.get("arguments", []))]
if not links:
    sys.exit("ERROR: no libonedal_dpc link action found for //cpp/oneapi/dal:dynamic_dpc; "
             "mnemonics seen: {}".format(sorted({a.get("mnemonic") for a in actions})))
for action in links:
    if "-Xscoverage" not in action.get("arguments", []):
        sys.exit("ERROR: -Xscoverage missing from the released DPC++ library link: {}"
                 .format(action.get("arguments", [])))
print("dynamic_dpc link action carries -Xscoverage")
PY

echo "code-coverage DPC++ smoke checks passed"
