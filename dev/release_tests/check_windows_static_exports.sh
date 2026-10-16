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

# Checks that oneDAL's Windows static libraries behave as plain archives: they carry no /EXPORT
# directives, and an EXE or a DLL linked with them exports nothing of oneDAL and needs no oneDAL
# DLL. Run from MSYS2 bash with the Visual Studio, MKL and TBB environments set up.
#
#   check_windows_static_exports.sh <release dir> [<build dir>]

set -euo pipefail

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

release=$(cd "$1/daal/latest" && pwd)
build=${2:-$(mktemp -d)}
src=$(cd "$(dirname "$0")/windows_static_exports" && pwd)
libdir=$release/lib/intel64

for tool in cl dumpbin cmake nmake; do
    command -v "$tool" > /dev/null || fail "$tool is not on PATH"
done

# Options are spelled with '-': MSYS2 would rewrite '/directives' as a path
count_matches() {
    awk -v pattern="$1" 'tolower($0) ~ pattern { n++ } END { print n + 0 }'
}

for lib in onedal_core.lib onedal.lib onedal_parameters.lib; do
    [ -f "$libdir/$lib" ] || continue
    exports=$(dumpbin -nologo -directives "$libdir/$lib" | count_matches '[-/]export:')
    echo "$lib: $exports /EXPORT directives"
    [ "$exports" -eq 0 ] || fail "$lib carries $exports /EXPORT directives"
done

tbb_dir=()
if [ -n "${TBBROOT:-}" ] && [ -d "$TBBROOT/lib/cmake/tbb" ]; then
    tbb_dir=(-DTBB_DIR="$TBBROOT/lib/cmake/tbb")
fi
cmake -S "$src" -B "$build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DoneDAL_DIR="$libdir/cmake/oneDAL" "${tbb_dir[@]}"
cmake --build "$build"

for binary in consumer.exe plugin.dll; do
    path=$build/$binary
    size=$(stat -c %s "$path")
    # dumpbin lists one 'ordinal hint RVA name' row per export
    exports=$(dumpbin -nologo -exports "$path" | awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9A-F]+$/ && NF >= 4 { print $4 }')
    count=$(printf '%s' "$exports" | count_matches '.')
    kept=$(count_matches '@daal@@|@dal@oneapi@@' < "$build/${binary%.*}.map")
    echo "$binary: $size bytes, $count exports, $kept oneDAL symbols in the link map"
    if [ "$binary" = plugin.dll ]; then
        [ "$exports" = plugin_compute_means ] || fail "plugin.dll exports: $exports"
    else
        [ "$count" -eq 0 ] || fail "consumer.exe exports: $exports"
        [ ! -e "$build/consumer.exp" ] && [ ! -e "$build/consumer.lib" ] ||
            fail "the linker wrote an import library for consumer.exe"
    fi
    dependents=$(dumpbin -nologo -dependents "$path" | count_matches 'onedal')
    [ "$dependents" -eq 0 ] || fail "$binary depends on a oneDAL DLL"
done

export PATH="$build:$release/../../tbb/latest/bin/vc_mt:$PATH"
"$build/consumer.exe" || fail "consumer.exe returned $?"
"$build/plugin_host.exe" || fail "plugin_host.exe returned $?"
echo "static linking checks passed"
