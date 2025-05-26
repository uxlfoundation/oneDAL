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

VERSION=v3.3.0
UNPACKED=ec-linux-amd64
ASSET=$UNPACKED.tar.gz
CHECKSUMS=checksums.txt
BASE_LINK=https://github.com/editorconfig-checker/editorconfig-checker/releases/download/$VERSION

# Download asset
wget $BASE_LINK/$ASSET

# Download checksum file
wget $BASE_LINK/$CHECKSUMS

# Verify checksum file
if ! grep -E "$ASSET$" $CHECKSUMS | sha256sum --check; then
    echo "Checksum verification failed"
    exit 1
fi

# Install
mkdir $UNPACKED && tar -xzf "$ASSET" -C $UNPACKED
mv $UNPACKED/bin/$UNPACKED /usr/local/bin/editorconfig-checker

# Clean up the downloaded files
rm -rf "$UNPACKED" "$ASSET" "$CHECKSUMS"
