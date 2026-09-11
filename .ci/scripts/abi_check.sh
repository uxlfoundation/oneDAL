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

# Cross-flavor public-symbol drift report.
#
# abidiff above catches per-library ABI changes. When debug info is stripped
# (the DPC++ libraries in CI to keep memory usage in check, see
# https://github.com/uxlfoundation/oneDAL/pull/3703), abidiff falls back to
# symbol addition/removal and misses layout/return-type breaks. As a cheap
# supplement, diff the set of newly-exported symbols between the host (_c)
# and DPC++ flavors of the same library. A symbol added only to the DPC++
# flavor is public surface that abidiff cannot type-check, so it is reported
# for manual review. This works on .dynsym alone, so debug info is not
# required.
#
# This report is deliberately non-fatal. The DPC++ library compiles every
# host translation unit plus the *_dpc.cpp ones (see ONEAPI.srcs.dpc in the
# top-level makefile), so its symbol set is structurally a superset: every
# PR that adds a GPU implementation legitimately adds DPC++-only symbols.
# Failing here would force such PRs to carry the 'API/ABI breaking change'
# label, which skips the whole job (.github/workflows/ci.yml) and would
# therefore disable the abidiff gate above to silence an expected message.
if ! command -v nm >/dev/null 2>&1; then
    echo "::error:: nm not found (required for cross-flavor symbol drift report)"
    exit 1
fi

# Namespaces that oneDAL does not treat as part of the public ABI. Kept in
# sync with the [suppress_*] entries of .github/.abignore so that this report
# and abidiff agree on what "public" means; update both together.
internal_sym_re='_Z+.*4daal[[:digit:]].*8internal[[:digit:]].*|_Z+N6oneapi3dal[[:digit:]].*(7backend|6detail|7preview)[[:digit:]].*'

public_syms () {
    # Sorted, unique names of the defined external dynamic symbols of $1,
    # minus the internal namespaces above.
    local dump
    # nm is called on its own so a read failure propagates: an ABI report that
    # silently degrades to an empty symbol set would claim "no drift" on tool
    # failure. Note that pipefail cannot do this job here, because a failure
    # inside a process substitution is invisible to the enclosing pipeline.
    dump=$(nm -D --defined-only --extern-only "$1") || return 1
    # filtering in awk (not grep) keeps the pipeline status meaningful: grep
    # exits 1 when every symbol happens to be filtered out.
    printf '%s\n' "$dump" | awk -v re="$internal_sym_re" 'NF && $NF !~ re { print $NF }' | sort -u
}

new_syms () {
    # public symbols present in $2 but not in $1.
    local before after
    before=$(public_syms "$1") || return 1
    after=$(public_syms "$2") || return 1
    # printf '%s' (not echo) so an empty set doesn't inject a spurious blank
    # line into comm's input, which would produce false-positive drift.
    comm -13 <(printf '%s' "$before") <(printf '%s' "$after")
}

# report at most this many symbol names per library, to keep the log readable
max_reported_syms=50

report_syms () {
    # $1: header line, $2: newline separated symbol list (non-empty)
    local count
    count=$(printf '%s\n' "$2" | wc -l)
    echo "::warning:: ${count} $1"
    printf '%s\n' "$2" | head -n "${max_reported_syms}"
    if [ "${count}" -gt "${max_reported_syms}" ]; then
        echo "... (${count} total, truncated to ${max_reported_syms})"
    fi
}

pairs=(
    "libonedal.so:libonedal_dpc.so"
    "libonedal_parameters.so:libonedal_parameters_dpc.so"
)

for pair in "${pairs[@]}"; do
    host_lib=${pair%:*}
    dpc_lib=${pair#*:}
    echo "======== cross-flavor symbol drift: ${host_lib} vs ${dpc_lib} ========"
    if [ ! -f "$main_release_dir/$host_lib" ] || [ ! -f "$release_dir/$host_lib" ] || \
       [ ! -f "$main_release_dir/$dpc_lib" ]  || [ ! -f "$release_dir/$dpc_lib" ]; then
        # not an error: the host-only build configurations have no _dpc libraries
        echo "skipped: ${host_lib} and/or ${dpc_lib} not present in both builds"
        continue
    fi
    if ! host_new=$(new_syms "$main_release_dir/$host_lib" "$release_dir/$host_lib"); then
        echo "::error:: nm failed for ${host_lib}"
        RETURN_CODE=$((RETURN_CODE+1))
        continue
    fi
    if ! dpc_new=$(new_syms "$main_release_dir/$dpc_lib" "$release_dir/$dpc_lib"); then
        echo "::error:: nm failed for ${dpc_lib}"
        RETURN_CODE=$((RETURN_CODE+1))
        continue
    fi
    only_in_dpc=$(comm -13 <(printf '%s' "$host_new") <(printf '%s' "$dpc_new"))
    only_in_host=$(comm -23 <(printf '%s' "$host_new") <(printf '%s' "$dpc_new"))
    if [ -n "$only_in_dpc" ]; then
        report_syms "new public symbol(s) in ${dpc_lib} with no counterpart in ${host_lib}. \
These are only checked for addition/removal, as ${dpc_lib} is built without debug info; \
review any layout or signature change to them by hand." "$only_in_dpc"
    fi
    if [ -n "$only_in_host" ]; then
        report_syms "new public symbol(s) in ${host_lib} with no counterpart in ${dpc_lib}. \
The DPC++ library is built from a superset of the host sources, so this is unexpected: \
check whether a declaration is guarded on ONEDAL_DATA_PARALLEL by mistake." "$only_in_host"
    fi
    if [ -z "$only_in_dpc" ] && [ -z "$only_in_host" ]; then
        echo "no drift"
    fi
done

exit ${RETURN_CODE}
