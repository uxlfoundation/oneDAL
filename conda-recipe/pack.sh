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

RELEASE_DIR=$(find . -maxdepth 1 -name "__release_*" -type d | head -1)
cd "$RELEASE_DIR/daal/latest"

mkdir -p "$PREFIX/lib"
# copy devel content
if [ "$PKG_NAME" = "dal-devel" ]; then
    cp -r "lib/cmake" "$PREFIX/lib/"
    cp -r "lib/pkgconfig" "$PREFIX/lib/"
fi
# copy headers
if [ "$PKG_NAME" = "dal-include" ]; then
    mkdir -p "$PREFIX/include"
    cp -r include/* "$PREFIX/include/"
fi
# copy libraries
if [ "$PKG_NAME" = "dal" ]; then
    find lib/intel64 -name "libonedal*.so.3" -exec cp {} "$PREFIX/lib/" \;
fi
if [ "$PKG_NAME" = "dal-static" ]; then
    find lib/intel64 -name "libonedal*.a" -exec cp {} "$PREFIX/lib/" \;
fi
