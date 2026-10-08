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

# abidiff reads DWARF when it is there and the symbol-only comparisons below
# need nm; both are used unconditionally, so a missing binutils would otherwise
# surface as a wrong answer rather than as a failure.
for tool in nm objcopy readelf; do
    if [ -z "$(command -v "$tool")" ]; then
        echo "::error:: ${tool} not found (binutils is required, see .ci/env/apt.sh abigail)"
        exit 1
    fi
done

has_dwarf () {
    # $1: path to a shared library. Succeeds if it has a .debug_info section.
    # libabigail builds a type-aware corpus from .debug_info and an
    # ELF-symbol-only one without it, so this decides which kind of comparison
    # a given library can take part in.
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

    # The two sides come from different builds -- this branch against a cached
    # build of main -- so the commit that turns debug info on or off for a
    # library leaves them disagreeing about it until main is rebuilt. abidiff
    # does not fail on that (it falls back to the symbol sets), but the result is
    # a weaker check than the log claims, and nothing says so. Say so, and strip
    # both sides so the comparison is the same in either direction no matter how
    # libabigail chooses to mix a DWARF corpus with a symbols-only one.
    # A missing $new is reported by abidiff below; readelf only needs to see files.
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
        # With debug info present this is the common outcome for a real type
        # change, so surface it in the checks UI instead of leaving it to whoever
        # scrolls the log.
        echo "::warning:: ${name}: abidiff reports a sub-type change it considers" \
             "possibly compatible. Not a failure, but read the diff above."
    fi
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
#
# Namespaces that oneDAL does not treat as part of the public ABI. Derived from
# .github/.abignore rather than restated here, so this report and abidiff cannot
# drift apart on what "public" means: the suppression file is the single source.
#
# Only unconditional suppressions are taken, i.e. [suppress_{type,variable,
# function}] stanzas that say 'drop = yes' and carry no 'change_kind'. A stanza
# with a 'change_kind' says "this kind of change to these symbols is allowed",
# not "these symbols are internal" -- the added-{type,variable,function} block
# at the end of .abignore covers every oneapi::dal symbol, so honouring it here
# would filter the entire public surface away and the report would go blind.
#
# libabigail's symbol_name_regexp is POSIX ERE, the same dialect awk matches
# with, so the patterns transfer verbatim.
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

# An empty set would classify every symbol as public-and-unchanged-looking in
# one direction and swallow the whole surface in the other, so refuse to run a
# report whose notion of "internal" came out empty.
if [ -z "$internal_sym_re" ]; then
    echo "::error:: no unconditional [suppress_*] entries parsed from ${abignore}"
    exit 1
fi
echo "internal-namespace filter from ${abignore}: ${internal_sym_re}"

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
    # exits 1 when every symbol happens to be filtered out. The pattern is
    # passed through the environment rather than with `-v`, which would expand
    # escape sequences in it and silently widen a pattern that spells a literal
    # `\.` or `\+`.
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
