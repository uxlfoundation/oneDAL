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

VERSION=v3.1.2
NAME=ec-linux-amd64
ASSET=$NAME.tar.gz

# Download
export SHA256="fc698b0bf5bca0d42e28dd59d72e25487a51f645ca242c5f74bae975369f16aa  $ASSET"
wget https://github.com/editorconfig-checker/editorconfig-checker/releases/download/$VERSION/$ASSET
echo "${SHA256}" | sha256sum --check

# Install
tar -xzf $ASSET
mv bin/$NAME /usr/local/bin/editorconfig-checker

# Clean up the downloaded archive
rm $ASSET
