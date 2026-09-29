#! /bin/bash
#===============================================================================
# Copyright 2020 Intel Corporation
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

# This script checks and formats C/C++ source files using clang-format.
# It supports three modes of operation:
# - pre-commit mode, with staged files as arguments,
# - CI mode,         without arguments in a continuous integration environment, and
# - local mode,      when the script is run manually by a developer.

RETURN_CODE=0

# Major version of clang-format used in CI. Other versions may format the code differently.
REQUIRED_CLANG_FORMAT_MAJOR=20

# The first argument, if given, is the clang-format executable to use.
CLANG_FORMAT_CANDIDATES=("$1" "${CLANG_FORMAT_EXE}" clang-format-20 clang-format)

# The remaining arguments, if any, are the files staged for commit
# (passed by the pre-commit framework).
[ $# -gt 0 ] && shift
STAGED_FILES=("$@")

CLANG_FORMAT_EXE=
for candidate in "${CLANG_FORMAT_CANDIDATES[@]}"; do
    [ -n "${candidate}" ] || continue
    candidate_major=$("${candidate}" --version 2> /dev/null | grep -oP 'version \K[0-9]+' | head -n 1)
    if [ "${candidate_major}" = "${REQUIRED_CLANG_FORMAT_MAJOR}" ]; then
        CLANG_FORMAT_EXE="${candidate}"
        break
    elif [ -n "${candidate_major}" ]; then
        echo "Skipping ${candidate}: version ${candidate_major} found, version ${REQUIRED_CLANG_FORMAT_MAJOR} required."
    fi
done
if [ -z "${CLANG_FORMAT_EXE}" ]; then
    echo "clang-format version ${REQUIRED_CLANG_FORMAT_MAJOR} not found or not working properly."
    echo "Install clang-format-${REQUIRED_CLANG_FORMAT_MAJOR} or set CLANG_FORMAT_EXE to its path."
    exit 1
fi

SOURCES_PATHS=(cpp/daal cpp/oneapi examples/oneapi examples/daal samples/oneapi samples/daal)
SOURCES_EXTENSIONS_REGEX=".*\.(c|cpp|h|hpp|cl|i)$"

echo "Using clang-format version: $(${CLANG_FORMAT_EXE} --version)"
echo "Starting format check..."

if [ ${#STAGED_FILES[@]} -gt 0 ]; then
    # Pre-commit mode: format the staged files in place and fail if any of them
    # was changed by clang-format, i.e. was not formatted correctly.
    NOT_FORMATTED_FILES=()
    for filename in "${STAGED_FILES[@]}"; do
        [ -f "${filename}" ] || continue
        echo "${filename}" | grep -qP "${SOURCES_EXTENSIONS_REGEX}" || continue

        in_sources_path=0
        for sources_path in "${SOURCES_PATHS[@]}"; do
            case "${filename}" in
            "${sources_path}"/*) in_sources_path=1 ;;
            esac
        done
        [ ${in_sources_path} -eq 1 ] || continue

        hash_before=$(git hash-object "${filename}")
        if ! "${CLANG_FORMAT_EXE}" -style=file -i "${filename}"; then
            echo "clang-format failed for ${filename}"
            RETURN_CODE=1
            continue
        fi
        hash_after=$(git hash-object "${filename}")

        if [ "${hash_before}" != "${hash_after}" ]; then
            NOT_FORMATTED_FILES+=("${filename}")
        fi
    done

    if [ ${#NOT_FORMATTED_FILES[@]} -gt 0 ]; then
        echo "Clang-format check FAILED! Found not formatted files:"
        printf '    %s\n' "${NOT_FORMATTED_FILES[@]}"
        echo "The files were reformatted in place. Review the changes and stage them with 'git add'."
        RETURN_CODE=3
    elif [ ${RETURN_CODE} -eq 0 ]; then
        echo "Clang-format check PASSED! Not formatted files not found..."
    fi

    exit ${RETURN_CODE}
fi

# CI mode: format all the sources and expect no changes in the working tree.
for sources_path in "${SOURCES_PATHS[@]}"; do
    pushd ${sources_path} || exit 1
    for filename in $(find . -type f | grep -P "${SOURCES_EXTENSIONS_REGEX}"); do ${CLANG_FORMAT_EXE} -style=file -i "${filename}"; done

    # Only the changes inside the current sources path are taken into account.
    if [ -n "$(git status --porcelain -- .)" ]; then
        echo "Clang-format check FAILED for ${sources_path}! Found not formatted files!"
        git status -- .
        RETURN_CODE=3
    else
        echo "Clang-format check PASSED for ${sources_path}! Not formatted files not found..."
    fi
    popd || exit 1
done

exit ${RETURN_CODE}
