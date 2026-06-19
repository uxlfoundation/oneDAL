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

input=$1
output=$2
cpus=$3

if [ "${cpus}" == "" ]; then
    cp $input $output
    exit
fi

function join { local IFS="$1"; shift; echo "$*"; }

replacements=()
for cpu in $cpus
do
    replacements+=("^#define DAAL_KERNEL_${cpu^^}\b")
done

sed_args=$(join '|' "${replacements[@]}")
sed -E "s/${sed_args}//" $input > $output
