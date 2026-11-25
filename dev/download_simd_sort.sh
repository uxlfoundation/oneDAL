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

# Fetches the SIMD sort library release for an architecture, pinned in
# dev/bazel/deps/simd_sort.bzl, into __deps/simd_sort, where the Make build
# looks when SIMD_SORT_ROOT is unset.
#
# Usage: dev/download_simd_sort.sh [intel64]   (default: intel64)

set -euo pipefail

ARCH="${1:-intel64}"
ONEDAL_DIR=$(cd "$(dirname "$0")/.." && pwd)
PIN_FILE="${ONEDAL_DIR}/dev/bazel/deps/simd_sort.bzl"

version() {
    sed -n "s/^$1 = \"\(v[0-9][0-9.]*\)\"\$/\1/p" "${PIN_FILE}"
}

sha256_of() {
    sed -n "s/^ *\"$1\": \"\([0-9a-f]\{64\}\)\",\{0,1\}\$/\1/p" "${PIN_FILE}"
}

case "${ARCH}" in
    intel64)
        VERSION=$(version _X86_VERSION)
        URL="https://github.com/intel/x86-simd-sort/archive/refs/tags/${VERSION}.tar.gz"
        ;;
    *)
        echo "No SIMD sort library for ${ARCH}"
        exit 1
        ;;
esac

if [ -z "${VERSION}" ]; then
    echo "Cannot read the ${ARCH} version from ${PIN_FILE}"
    exit 1
fi
SHA256=$(sha256_of "${VERSION}")
if [ -z "${SHA256}" ]; then
    echo "${PIN_FILE} has no sha256 for ${VERSION}. Add it from:"
    echo "  curl -fsSL ${URL} | sha256sum"
    exit 1
fi

DEPS_DIR="${ONEDAL_DIR}/__deps"
DST="${DEPS_DIR}/simd_sort"
STAMP="${DST}/.pin"

if [ -f "${STAMP}" ] && [ "$(cat "${STAMP}")" = "${ARCH} ${VERSION} ${SHA256}" ]; then
    echo "SIMD sort for ${ARCH} ${VERSION} is already in ${DST}"
    exit 0
fi

mkdir -p "${DEPS_DIR}"
ARCHIVE="${DEPS_DIR}/simd_sort-${VERSION}.tar.gz"

if [ -x "$(command -v curl)" ]; then
    echo curl -fL -o "${ARCHIVE}" "${URL}"
    curl -fL -o "${ARCHIVE}" "${URL}"
elif [ -x "$(command -v wget)" ]; then
    echo wget -O "${ARCHIVE}" "${URL}"
    wget -O "${ARCHIVE}" "${URL}"
else
    echo "curl or wget not available"
    exit 1
fi

if [ -x "$(command -v sha256sum)" ]; then
    ACTUAL=$(sha256sum "${ARCHIVE}" | cut -d ' ' -f 1)
else
    ACTUAL=$(shasum -a 256 "${ARCHIVE}" | cut -d ' ' -f 1)
fi
if [ "${ACTUAL}" != "${SHA256}" ]; then
    echo "sha256 mismatch for ${URL}: expected ${SHA256}, got ${ACTUAL}"
    rm -f "${ARCHIVE}"
    exit 1
fi

rm -rf "${DST}"
mkdir -p "${DST}"
tar -C "${DST}" --strip-components=1 -xzf "${ARCHIVE}"
rm -f "${ARCHIVE}"
echo "${ARCH} ${VERSION} ${SHA256}" > "${STAMP}"
echo "SIMD sort for ${ARCH} ${VERSION} extracted to ${DST}"
