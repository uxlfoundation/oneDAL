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
# Tests for Bazel release structure: versioning, symlinks, vars.sh, pkg-config.
# Run after: bazel build //:release (which also builds env/vars.sh and pkg-config files)
#
# Usage:
#   ./dev/bazel/tests/release_structure_test.sh <release_dir>
#
# Example:
#   ./dev/bazel/tests/release_structure_test.sh bazel-bin/release/daal/latest

set -euo pipefail

RELEASE_DIR="${1:-bazel-bin/release/daal/latest}"
LIB_DIR="${RELEASE_DIR}/lib/intel64"
PASS=0
FAIL=0

_pass() { echo "  ✅ PASS: $1"; PASS=$((PASS + 1)); }
_fail() { echo "  ❌ FAIL: $1"; FAIL=$((FAIL + 1)); }

# ---------------------------------------------------------------------------
# 1. Library versioning and symlinks
# ---------------------------------------------------------------------------
echo ""
echo "=== Library versioning & symlinks ==="

for lib_base in libonedal_core libonedal libonedal_thread; do
    so="${LIB_DIR}/${lib_base}.so"
    # Find versioned file using bash globbing (more robust than ls+grep)
    shopt -s nullglob
    versioned_candidates=( "${LIB_DIR}/${lib_base}.so."[0-9]*.[0-9]* )
    shopt -u nullglob

    if [ "${#versioned_candidates[@]}" -eq 0 ]; then
        _fail "${lib_base}.so.X.Y not found in ${LIB_DIR}"
        continue
    elif [ "${#versioned_candidates[@]}" -gt 1 ]; then
        _fail "${lib_base}: multiple versioned files found: ${versioned_candidates[*]}"
        continue
    fi

    versioned="${versioned_candidates[0]}"

    # Check versioned file is a real file
    if [ -f "$versioned" ] && [ ! -L "$versioned" ]; then
        _pass "${lib_base}.so.X.Y is a real file"
    else
        _fail "${lib_base}.so.X.Y should be a regular file, not a symlink"
    fi

    # Extract major.minor from filename
    version_suffix="${versioned##*.so.}"  # e.g. "3.0"
    major_ver="${version_suffix%%.*}"     # e.g. "3"

    # Check major symlink exists and points correctly
    major_link="${LIB_DIR}/${lib_base}.so.${major_ver}"
    if [ -L "$major_link" ]; then
        target=$(readlink "$major_link")
        expected_target="$(basename "$versioned")"
        if [ "$target" = "$expected_target" ]; then
            _pass "${lib_base}.so.${major_ver} → ${expected_target}"
        else
            _fail "${lib_base}.so.${major_ver} points to $target, expected $expected_target"
        fi
    else
        _fail "${lib_base}.so.${major_ver} symlink missing"
    fi

    # Check unversioned symlink points to major link
    if [ -L "$so" ]; then
        target=$(readlink "$so")
        expected_target="${lib_base}.so.${major_ver}"
        if [ "$target" = "$expected_target" ]; then
            _pass "${lib_base}.so → ${expected_target}"
        else
            _fail "${lib_base}.so points to $target, expected $expected_target"
        fi
    else
        _fail "${lib_base}.so unversioned symlink missing"
    fi
done

# ---------------------------------------------------------------------------
# 2. SONAME check
# ---------------------------------------------------------------------------
echo ""
echo "=== SONAME check ==="

for lib in "${LIB_DIR}"/libonedal_core.so.*.* "${LIB_DIR}"/libonedal.so.*.* "${LIB_DIR}"/libonedal_thread.so.*.* ; do
    [ -f "$lib" ] || continue
    lib_base=$(basename "$lib" | sed 's/\.so\..*//')
    # Extract expected SONAME from filename: libonedal_core.so.3.0 → libonedal_core.so.3
    version_suffix="${lib##*.so.}"
    major_ver="${version_suffix%%.*}"
    expected_soname="${lib_base}.so.${major_ver}"

    soname=$(readelf -d "$lib" 2>/dev/null | grep SONAME | grep -oE '\[[^]]+\]' | tr -d '[]' || true)
    if [ -n "$soname" ]; then
        if [ "$soname" = "$expected_soname" ]; then
            _pass "${lib_base}: SONAME = ${soname} (correct)"
        else
            _fail "${lib_base}: SONAME = ${soname}, expected ${expected_soname}"
        fi
    else
        _fail "${lib_base}: SONAME not found in ${lib}"
    fi
done

# ---------------------------------------------------------------------------
# 3. vars.sh
# ---------------------------------------------------------------------------
echo ""
echo "=== vars.sh ==="

VARS_SH="${RELEASE_DIR}/env/vars.sh"
if [ -f "$VARS_SH" ]; then
    _pass "env/vars.sh exists"
    if grep -q "DAL_MAJOR_BINARY" "$VARS_SH"; then
        _pass "vars.sh contains DAL_MAJOR_BINARY"
    else
        _fail "vars.sh missing DAL_MAJOR_BINARY"
    fi
    # Check no unresolved placeholders
    if grep -q "__DAL_MAJOR_BINARY__\|__DAL_MINOR_BINARY__" "$VARS_SH"; then
        _fail "vars.sh still has unresolved placeholders"
    else
        _pass "vars.sh has no unresolved placeholders"
    fi
else
    _fail "env/vars.sh not found at ${VARS_SH}"
fi

# ---------------------------------------------------------------------------
# 4. pkg-config
# ---------------------------------------------------------------------------
echo ""
echo "=== pkg-config ==="

PC_FILE="${RELEASE_DIR}/lib/pkgconfig/onedal.pc"
if [ -f "$PC_FILE" ]; then
    _pass "lib/pkgconfig/onedal.pc exists"
    if grep -q "^Name:" "$PC_FILE"; then
        _pass "onedal.pc has Name field"
    else
        _fail "onedal.pc missing Name field"
    fi
    if grep -q "^Libs:" "$PC_FILE"; then
        _pass "onedal.pc has Libs field"
    else
        _fail "onedal.pc missing Libs field"
    fi
else
    _fail "lib/pkgconfig/onedal.pc not found"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

[ "$FAIL" -eq 0 ] || exit 1
