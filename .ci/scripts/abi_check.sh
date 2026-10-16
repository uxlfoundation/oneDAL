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

# A missing binutils would otherwise give a wrong answer instead of a failure.
for tool in nm objcopy readelf; do
    if [ -z "$(command -v "$tool")" ]; then
        echo "::error:: ${tool} not found (binutils is required, see .ci/env/apt.sh abigail)"
        exit 1
    fi
done

has_dwarf () {
    # $1: path to a shared library. Succeeds if it has a .debug_info section.
    readelf -SW "$1" | grep -q '[[:space:]]\.debug_info[[:space:]]'
}

strip_dir=$(mktemp -d)
trap 'rm -rf "${strip_dir}"' EXIT

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
    old=$i
    new=$release_dir/$name

    # The cached main build may differ from this branch in having debug info.
    # abidiff then silently falls back to symbols only, so warn and compare
    # stripped copies of both. A missing $new is reported by abidiff below.
    dwarf_old=no; dwarf_new=no
    if [ -f "$new" ]; then
        has_dwarf "$old" && dwarf_old=yes
        has_dwarf "$new" && dwarf_new=yes
    fi
    if [ "$dwarf_old" != "$dwarf_new" ]; then
        echo "::warning:: ${name}: debug info present in only one build" \
             "(main: ${dwarf_old}, this branch: ${dwarf_new}). Comparing stripped" \
             "copies, so this library is checked for symbol addition and removal" \
             "only. Rerun the 'CI' workflow on main to restore the type-level check."
        if objcopy --strip-debug "$old" "${strip_dir}/main_${name}" &&
           objcopy --strip-debug "$new" "${strip_dir}/branch_${name}"; then
            old=${strip_dir}/main_${name}
            new=${strip_dir}/branch_${name}
        else
            echo "::error:: objcopy failed for ${name}; comparing unstripped"
            RETURN_CODE=$((RETURN_CODE+1))
        fi
    fi

    abidiff --suppr .github/.abignore "$old" "$new"
    retVal=$?
    # ignore a return value of 4 as it signifies a possibly compatible change
    if [ $retVal != 4 ]; then
        RETURN_CODE=$(($RETURN_CODE+$retVal))
    else
        # With debug info this is the usual result of a real type change.
        echo "::warning:: ${name}: abidiff reports a sub-type change it considers" \
             "possibly compatible. Not a failure, but read the diff above."
    fi
done

# Cross-flavor public-symbol drift report.
#
# The DPC++ libraries are built without debug info, so abidiff only checks
# their symbols for addition and removal. Report public symbols added to a
# DPC++ library but not to its host library: that surface is not type-checked.
# Warnings only, since every new GPU implementation adds such symbols.
#
# Internal namespaces come from the unconditional suppressions in .abignore
# ('drop = yes', no 'change_kind'). The change_kind stanzas cover all of
# oneapi::dal, so taking them would filter out the whole public surface.
# symbol_name_regexp is POSIX ERE, so awk can use the patterns verbatim.
abignore=.github/.abignore
if [ ! -f "$abignore" ]; then
    echo "::error:: ${abignore} not found (required to know which namespaces are internal)"
    exit 1
fi

internal_sym_re=$(awk '
    function flush() {
        if (suppress && drop == "yes" && change_kind == "" && re != "" && !(re in seen))
        {
            seen[re] = 1
            out = (out == "" ? re : out "|" re)
        }
        suppress = 0; drop = ""; change_kind = ""; re = ""
    }
    { sub(/^[[:space:]]+/, ""); sub(/[[:space:]]+$/, "") }
    /^;/ || /^$/ { next }
    /^\[/ {
        flush()
        suppress = ($0 ~ /^\[suppress_(type|variable|function)\][[:space:]]*$/)
        next
    }
    {
        key = $0; sub(/[[:space:]]*=.*$/, "", key)
        val = $0; sub(/^[^=]*=[[:space:]]*/, "", val)
        if (key == "symbol_name_regexp") re = val
        else if (key == "drop") drop = val
        else if (key == "change_kind") change_kind = val
    }
    END { flush(); print out }
' "$abignore")

# An empty filter means the parser no longer understands .abignore.
if [ -z "$internal_sym_re" ]; then
    echo "::error:: no unconditional [suppress_*] entries parsed from ${abignore}"
    exit 1
fi
echo "internal-namespace filter from ${abignore}: ${internal_sym_re}"

public_syms () {
    # Sorted, unique names of the defined external dynamic symbols of $1,
    # minus the internal namespaces above.
    local dump
    # nm runs on its own so that its failure is not reported as "no drift".
    dump=$(nm -D --defined-only --extern-only "$1") || return 1
    # awk, not grep, which exits 1 when everything is filtered out. The pattern
    # goes through ENVIRON because `awk -v` would expand its backslashes.
    printf '%s\n' "$dump" |
        internal_sym_re="$internal_sym_re" \
            awk 'NF && $NF !~ ENVIRON["internal_sym_re"] { print $NF }' |
        sort -u
}

new_syms () {
    # public symbols present in $2 but not in $1.
    local before after
    before=$(public_syms "$1") || return 1
    after=$(public_syms "$2") || return 1
    # printf '%s', not echo: an empty set must not feed comm a blank line.
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
