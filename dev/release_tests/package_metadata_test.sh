#!/usr/bin/env sh
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

set -eu

ROOT_DIR="${1:?usage: package_metadata_test.sh <release-or-install-prefix>}"
PASS=0
FAIL=0

OS="$(uname -s)"

pass() {
    echo "  PASS: $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "  FAIL: $1"
    FAIL=$((FAIL + 1))
}

check_pkg_config() {
    pc_name="$1"
    pc_file="${ROOT_DIR}/lib/pkgconfig/${pc_name}"

    if [ -f "$pc_file" ]; then
        pass "lib/pkgconfig/${pc_name} exists"
    else
        fail "lib/pkgconfig/${pc_name} not found"
        return
    fi

    if grep -q "^Name:" "$pc_file"; then
        pass "${pc_name} has Name field"
    else
        fail "${pc_name} missing Name field"
    fi

    if grep -q "^Libs:" "$pc_file"; then
        pass "${pc_name} has Libs field"
    else
        fail "${pc_name} missing Libs field"
    fi
}

echo ""
echo "=== package metadata ==="

vars_file="${ROOT_DIR}/env/vars.sh"
if [ -f "$vars_file" ]; then
    pass "env/vars.sh exists"
    if grep -q "DAL_MAJOR_BINARY" "$vars_file"; then
        pass "vars.sh contains DAL_MAJOR_BINARY"
    else
        fail "vars.sh missing DAL_MAJOR_BINARY"
    fi
    if grep -q "__DAL_MAJOR_BINARY__\|__DAL_MINOR_BINARY__" "$vars_file"; then
        fail "vars.sh still has unresolved placeholders"
    else
        pass "vars.sh has no unresolved placeholders"
    fi
else
    fail "env/vars.sh not found at ${vars_file}"
fi

if [ "$OS" = "Linux" ]; then
    check_pkg_config "dal-dynamic-threading-host.pc"
    check_pkg_config "dal-static-threading-host.pc"
else
    check_pkg_config "onedal.pc"
fi

echo "=== package metadata results: ${PASS} passed, ${FAIL} failed ==="

[ "$FAIL" -eq 0 ] || exit 1
