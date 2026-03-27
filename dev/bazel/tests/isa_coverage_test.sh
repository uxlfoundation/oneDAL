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
#
# Verifies that libonedal_core.so contains CPU dispatch symbols for all
# required ISA variants: sse2, sse42, avx2, avx512 (CpuType E0/E2/E4/E6).
#
# Make builds all 4 ISA variants by default. Building via //:release
# automatically compiles all ISA variants (cfg transition on release rule).
# This test enforces that contract.

set -euo pipefail

LIB="${1:-}"
if [[ -z "${LIB}" ]]; then
    echo "Usage: $0 <path-to-libonedal_core.so>"
    exit 1
fi

if [[ ! -f "${LIB}" ]]; then
    echo "ERROR: Library not found: ${LIB}"
    exit 1
fi

PASS=0
FAIL=0

check_isa() {
    local cpu_type="$1"   # e.g. CpuTypeE2
    local isa_name="$2"   # e.g. sse42
    local count
    count=$(nm -D "${LIB}" | grep -c "${cpu_type}" || true)
    if [[ "${count}" -gt 0 ]]; then
        echo "  OK  ${isa_name} (${cpu_type}): ${count} dispatch symbols"
        PASS=$((PASS + 1))
    else
        echo "  FAIL ${isa_name} (${cpu_type}): 0 dispatch symbols found"
        echo "       Build via //:release (uses all ISAs via cfg transition) or pass --cpu=all"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== ISA coverage check: $(basename "${LIB}") ==="
check_isa "CpuTypeE0" "sse2"
check_isa "CpuTypeE2" "sse42"
check_isa "CpuTypeE4" "avx2"
check_isa "CpuTypeE6" "avx512"

echo ""
if [[ "${FAIL}" -gt 0 ]]; then
    echo "RESULT: FAILED (${FAIL} ISA(s) missing, ${PASS} present)"
    exit 1
else
    echo "RESULT: PASSED (all 4 ISA variants present)"
    exit 0
fi
