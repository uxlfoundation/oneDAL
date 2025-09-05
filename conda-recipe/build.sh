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

export TBBROOT=$PREFIX
export LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH}"

# oneDAL build system requires non-versioned library names, conda-forge::tbb doesn't provide them
ln -s $PREFIX/lib/libtbb.so.12 $PREFIX/lib/libtbb.so
ln -s $PREFIX/lib/libtbbmalloc.so.2 $PREFIX/lib/libtbbmalloc.so

# flags set by conda-build create problems with oneDAL build system
unset CFLAGS LDFLAGS CXXFLAGS
make -j$(nproc) daal oneapi 

RELEASE_DIR=$(find . -maxdepth 1 -name "__release_*" -type d | head -1)
cd "$RELEASE_DIR/daal/latest"

# copy headers
mkdir -p "$PREFIX/include"
cp -r include/* "$PREFIX/include/"

# copy libraries
mkdir -p "$PREFIX/lib"
find lib/intel64 -name "libonedal*.so.3" -exec cp {} "$PREFIX/lib/" \;
find lib/intel64 -name "libonedal*.a" -exec cp {} "$PREFIX/lib/" \; 2>/dev/null || true
