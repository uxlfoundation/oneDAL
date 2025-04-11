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
    abidiff $i $release_dir/$name
    retVal=$?
    # ignore a return value of 4 as it signifies a possibly compatible change
    if [ $retVal != 4 ]; then RETURN_CODE=$(($RETURN_CODE+$retVal)); fi
done
exit ${RETURN_CODE}
