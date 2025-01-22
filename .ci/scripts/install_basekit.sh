#!/bin/bash
#===============================================================================
# Copyright 2025 Intel Corporation
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

URL=$1
COMPONENTS=$2

curl --output webimage.sh --url "$URL" --retry 5 --retry-delay 5
chmod +x webimage.sh
./webimage.sh -x -f webimage_extracted --log extract.log
rm -rf webimage.sh
WEBIMAGE_NAME=$(ls -1 webimage_extracted/)
if [ -z "$COMPONENTS" ]; then
  sudo webimage_extracted/"$WEBIMAGE_NAME"/bootstrapper -s --action install --eula=accept --log-dir=.
  installer_exit_code=$?
else
  sudo webimage_extracted/"$WEBIMAGE_NAME"/bootstrapper -s --action install --components="$COMPONENTS" --eula=accept --log-dir=.
  installer_exit_code=$?
fi
rm -rf webimage_extracted
exit $installer_exit_code