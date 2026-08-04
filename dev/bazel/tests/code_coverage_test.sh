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

# Positive-path check for --code_coverage=true on the icx toolchain: builds
# DAAL core and confirms the actual compile actions carry the Make-equivalent
# coverage flags, and that the separately built threading module does not.
set -euo pipefail

bazel_cmd="${BAZEL:-bazel}"
startup_args=()
if [[ -n "${BAZEL_OUTPUT_USER_ROOT:-}" ]]; then
    startup_args+=("--output_user_root=${BAZEL_OUTPUT_USER_ROOT}")
fi
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

"${bazel_cmd}" "${startup_args[@]}" build //cpp/daal:core_static --code_coverage=true

"${bazel_cmd}" "${startup_args[@]}" aquery 'mnemonic("CppCompile", deps(//cpp/daal:core_static))' \
    --code_coverage=true --output=jsonproto >"${work}/core_actions.json"
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

"${bazel_cmd}" "${startup_args[@]}" build //cpp/daal:thread_static --code_coverage=true
"${bazel_cmd}" "${startup_args[@]}" aquery 'mnemonic("CppCompile", deps(//cpp/daal:thread_static))' \
    --code_coverage=true --output=jsonproto >"${work}/thread_actions.json"
python3 -c "
import json, sys
with open('${work}/thread_actions.json') as f:
    data = json.load(f)
for action in data.get('actions', []):
    if '-DGCOV_BUILD' in action.get('arguments', []):
        sys.exit('ERROR: threading_tbb compile action unexpectedly defines GCOV_BUILD')
print('threading_tbb compile actions do not define GCOV_BUILD')
"

echo "code-coverage icx smoke checks passed"
