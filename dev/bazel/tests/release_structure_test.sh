#!/usr/bin/env bash
# Tests for Bazel release structure: versioning, symlinks, vars.sh, pkg-config.
# Run after: bazel build //:release :release_vars_sh :release_pkgconfig
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
    # Find actual versioned file
    versioned=$(ls "${LIB_DIR}/${lib_base}.so."* 2>/dev/null | grep -v ".so.[0-9]*$" || true)

    if [ -z "$versioned" ]; then
        _fail "${lib_base}.so.X.Y not found in ${LIB_DIR}"
        continue
    fi

    # Check versioned file is a real file
    if [ -f "$versioned" ] && [ ! -L "$versioned" ]; then
        _pass "${lib_base}.so.X.Y is a real file"
    else
        _fail "${lib_base}.so.X.Y should be a regular file, not a symlink"
    fi

    # Check major symlink exists and points correctly
    major_ver=$(echo "$versioned" | grep -oP '\.\d+\.\d+$' | cut -d. -f2)
    major_link="${LIB_DIR}/${lib_base}.so.${major_ver}"
    if [ -L "$major_link" ]; then
        _pass "${lib_base}.so.${major_ver} symlink exists"
    else
        _fail "${lib_base}.so.${major_ver} symlink missing"
    fi

    # Check unversioned symlink
    if [ -L "$so" ]; then
        _pass "${lib_base}.so unversioned symlink exists"
    else
        _fail "${lib_base}.so unversioned symlink missing"
    fi
done

# ---------------------------------------------------------------------------
# 2. SONAME check
# ---------------------------------------------------------------------------
echo ""
echo "=== SONAME check ==="

for lib in "${LIB_DIR}"/libonedal_core.so.*.* "${LIB_DIR}"/libonedal.so.*.* ; do
    [ -f "$lib" ] || continue
    lib_base=$(basename "$lib" | sed 's/\.so\..*//')
    soname=$(readelf -d "$lib" 2>/dev/null | grep SONAME | grep -oP '\[\K[^\]]+' || true)
    if [ -n "$soname" ]; then
        _pass "${lib_base}: SONAME = ${soname}"
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
