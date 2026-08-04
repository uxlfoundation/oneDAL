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

# DPC++ coverage is intentionally a nightly check: it validates the special
# linker driver option without making public PR CI install the full runtime.
set -euo pipefail

bazel_cmd="${BAZEL:-bazel}"
startup_args=()
if [[ -n "${BAZEL_OUTPUT_USER_ROOT:-}" ]]; then
    startup_args+=("--output_user_root=${BAZEL_OUTPUT_USER_ROOT}")
fi
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

target="//cpp/oneapi/dal/table:table_dpc"
"${bazel_cmd}" "${startup_args[@]}" build "${target}" \
    --code_coverage=true --release_dpc=true
"${bazel_cmd}" "${startup_args[@]}" aquery "deps(${target})" \
    --code_coverage=true --release_dpc=true --output=jsonproto >"${work}/actions.json"

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

echo "code-coverage DPC++ nightly smoke check passed"
