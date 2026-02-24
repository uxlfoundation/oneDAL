#!/usr/bin/env bash
#===============================================================================
# Copyright UXL Contribution to oneDAL project
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

# Test: MKL linkage parity with Make build
#
# Verifies that:
# 1. Static lib (libonedal_core.a) does NOT contain MKL archive objects
# 2. Dynamic lib (libonedal_core.so) does NOT export MKL symbols
#
# Only runs with MKL backend (target_compatible_with guards OpenBLAS).

set -euo pipefail

STATIC_LIB="${TEST_SRCDIR}/onedal/cpp/daal/libonedal_core.a"
DYNAMIC_LIB="${TEST_SRCDIR}/onedal/cpp/daal/libonedal_core.so"

PASS=0
FAIL=0

check() {
    local description="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "  PASS: $description"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $description"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== MKL Linkage Parity Test ==="
echo ""

# --- Static lib checks ---
echo "Static lib: $STATIC_LIB"
if [ ! -f "$STATIC_LIB" ]; then
    echo "ERROR: Static lib not found: $STATIC_LIB"
    exit 1
fi

# Static lib must NOT contain MKL archive object files
# (libmkl_core.a, libmkl_intel_ilp64.a, libmkl_tbb_thread.a objects)
MKL_ARCHIVE_OBJS=$(ar t "$STATIC_LIB" 2>/dev/null | grep -c '^_mkl_\|^mkl_blas\|^mkl_vml\|libmkl' || true)
check "Static lib: no MKL archive objects merged in" "$MKL_ARCHIVE_OBJS"

# Static lib must NOT have defined MKL symbols (only undefined refs are ok)
MKL_DEFINED=$(nm "$STATIC_LIB" 2>/dev/null | grep -c ' [Tt] mkl_' || true)
check "Static lib: no defined MKL symbols (U refs are ok)" "$MKL_DEFINED"

echo ""

# --- Dynamic lib checks ---
echo "Dynamic lib: $DYNAMIC_LIB"
if [ ! -f "$DYNAMIC_LIB" ]; then
    echo "ERROR: Dynamic lib not found: $DYNAMIC_LIB"
    exit 1
fi

# Dynamic lib must NOT export MKL symbols (they must be hidden)
MKL_EXPORTED=$(nm -D "$DYNAMIC_LIB" 2>/dev/null | grep -c ' mkl_' || true)
check "Dynamic lib: no exported MKL symbols (symbols hidden via --exclude-libs)" "$MKL_EXPORTED"

echo ""
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Expected behavior (Make parity):"
    echo "  - libonedal_core.a: MKL not embedded, user links MKL separately"
    echo "  - libonedal_core.so: MKL embedded from static archives, symbols hidden"
    exit 1
fi

exit 0
