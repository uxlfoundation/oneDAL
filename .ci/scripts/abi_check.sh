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

ci_dir=$(dirname $(dirname $(dirname "${BASH_SOURCE[0]}")))
cd $ci_dir

# relative paths must be made from the oneDAL repo root
main_release_dir=$1
release_dir=$2
RETURN_CODE=0

echo "Shared Library ABI Conformance"
solibs=($(ls $main_release_dir/lib*.so))
# if no .so files found to compare against, throw error
if [ ${#solibs[@]} -eq 0 ]; then
    echo "::error:: No shared objects found"
    exit 1
fi

for i in "${solibs[@]}"
do
    name=$(basename $i)
    echo "======== ${name} ========"
    abidiff --suppr .github/.abignore $i $release_dir/$name
    retVal=$?
    # ignore a return value of 4 as it signifies a possibly compatible change
    if [ $retVal != 4 ]; then RETURN_CODE=$(($RETURN_CODE+$retVal)); fi
done

# Cross-flavor public-symbol drift check.
#
# abidiff above catches per-library ABI changes. When debug info is stripped
# (the DPC++ libraries in CI to keep memory usage in check), abidiff falls
# back to symbol addition/removal and misses layout/return-type breaks. As a
# cheap supplement, diff the set of newly-exported symbols between the host
# (_c) and DPC++ flavors of the same library: if a PR adds a public symbol
# to one flavor but not the other, the public surface has diverged and a
# reviewer should look at it. This works on .dynsym alone, so debug info is
# not required.
if ! command -v nm >/dev/null 2>&1; then
    echo "::error:: nm not found (required for cross-flavor symbol drift check)"
    exit 1
fi

new_syms () {
    # symbols present in $2 but not in $1, filtered to defined + external.
    # nm errors are not suppressed: an ABI gate that silently degrades to an
    # empty symbol set would report "no drift" on tool failure.
    set -o pipefail
    comm -13 \
        <(nm -D --defined-only --extern-only "$1" | awk '{print $NF}' | sort -u) \
        <(nm -D --defined-only --extern-only "$2" | awk '{print $NF}' | sort -u)
}

pairs=(
    "libonedal.so:libonedal_dpc.so"
    "libonedal_parameters.so:libonedal_parameters_dpc.so"
)

for pair in "${pairs[@]}"; do
    host_lib=${pair%:*}
    dpc_lib=${pair#*:}
    if [ ! -f "$main_release_dir/$host_lib" ] || [ ! -f "$release_dir/$host_lib" ] || \
       [ ! -f "$main_release_dir/$dpc_lib" ]  || [ ! -f "$release_dir/$dpc_lib" ]; then
        continue
    fi
    echo "======== cross-flavor symbol drift: ${host_lib} vs ${dpc_lib} ========"
    if ! host_new=$(new_syms "$main_release_dir/$host_lib" "$release_dir/$host_lib"); then
        echo "::error:: nm/comm failed for ${host_lib}"
        RETURN_CODE=$((RETURN_CODE+1))
        continue
    fi
    if ! dpc_new=$(new_syms "$main_release_dir/$dpc_lib" "$release_dir/$dpc_lib"); then
        echo "::error:: nm/comm failed for ${dpc_lib}"
        RETURN_CODE=$((RETURN_CODE+1))
        continue
    fi
    # printf '%s' (not echo) so an empty set doesn't inject a spurious blank
    # line into comm's input, which would produce false-positive drift.
    only_in_dpc=$(comm -13 <(printf '%s' "$host_new") <(printf '%s' "$dpc_new"))
    only_in_host=$(comm -23 <(printf '%s' "$host_new") <(printf '%s' "$dpc_new"))
    if [ -n "$only_in_dpc" ]; then
        echo "::error:: new public symbols in ${dpc_lib} with no counterpart in ${host_lib}:"
        echo "$only_in_dpc"
        RETURN_CODE=$((RETURN_CODE+1))
    fi
    if [ -n "$only_in_host" ]; then
        echo "::error:: new public symbols in ${host_lib} with no counterpart in ${dpc_lib}:"
        echo "$only_in_host"
        RETURN_CODE=$((RETURN_CODE+1))
    fi
    if [ -z "$only_in_dpc" ] && [ -z "$only_in_host" ]; then
        echo "no drift"
    fi
done

exit ${RETURN_CODE}
