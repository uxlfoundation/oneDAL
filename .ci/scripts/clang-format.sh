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

RETURN_CODE=0

CLANG_FORMAT_EXE=${CLANG_FORMAT_EXE:-clang-format-20}

echo "Using clang-format version: $(${CLANG_FORMAT_EXE} --version)"
echo "Starting format check..."

for sources_path in cpp/daal cpp/oneapi examples/oneapi examples/daal samples/oneapi samples/daal; do
    pushd ${sources_path} || exit 1
    if find . -type f -regextype posix-extended -regex '.*\.(c|cpp|h|hpp|cl|i)' -print0 \
        | xargs -0 "${CLANG_FORMAT_EXE}" -style=file --dry-run --Werror; then
        echo "Clang-format check PASSED for ${sources_path}! Not formatted files not found..."
    else
        echo "Clang-format check FAILED for ${sources_path}! Found not formatted files!"
        RETURN_CODE=3
    fi
    popd || exit 1
done

exit ${RETURN_CODE}
