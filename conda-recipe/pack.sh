#!/usr/bin/env sh
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
    cp -r "env" "$PREFIX/"
    cp -r "lib/cmake" "$PREFIX/lib/"
    cp -r "lib/pkgconfig" "$PREFIX/lib/"
    # Install example datasets under $CONDA_PREFIX/share/oneDAL/data for dal-devel tests.
    # Prefer canonical source location ($SRC_DIR/data); keep release-layout fallbacks.
    if [ -d "$SRC_DIR/data" ]; then
        mkdir -p "$PREFIX/share/oneDAL/data"
        cp -f "$SRC_DIR/data/"*.csv "$PREFIX/share/oneDAL/data/" 2>/dev/null || true
    elif [ -d "$SRC_DIR/examples/oneapi/data" ]; then
        mkdir -p "$PREFIX/share/oneDAL/data"
        cp -f "$SRC_DIR/examples/oneapi/data/"*.csv "$PREFIX/share/oneDAL/data/" 2>/dev/null || true
    elif [ -d data ]; then
        mkdir -p "$PREFIX/share/oneDAL/data"
        cp -f data/*.csv "$PREFIX/share/oneDAL/data/" 2>/dev/null || true
    fi
    # set up links necessary for proper works of pkg-config, cmake and env. script
    mkdir -p "$PREFIX/lib/intel64"
    for lib in lib/intel64/libonedal*.so*; do
        # Keep dal-devel CPU-oriented: do not create intel64 links for DPC runtime libs.
        case "$lib" in
            *"_dpc"*) continue ;;
        esac
        if [ -f "$lib" ]; then
            libname=$(basename "$lib")
            ln -sf "../$libname" "$PREFIX/lib/intel64/$libname"
        fi
    done
    # WORKAROUND: empty file to force conda-build to include "lib/intel64" directory into devel package
    touch "$PREFIX/lib/intel64/.onedal_links_anchor"
fi
# copy headers
if [ "$PKG_NAME" = "dal-include" ]; then
    mkdir -p "$PREFIX/include"
    cp -r include/* "$PREFIX/include/"
fi
# copy CPU runtime libraries (excludes all DPC++ runtime libs, moved to dal-gpu)
if [ "$PKG_NAME" = "dal" ]; then
    find lib/intel64 -name "libonedal*.so*" ! -name "libonedal*_dpc*" -exec cp -P {} "$PREFIX/lib/" \;
fi
# copy GPU/DPC++ runtime libraries
if [ "$PKG_NAME" = "dal-gpu" ]; then
    find lib/intel64 -name "libonedal*_dpc*.so*" -exec cp -P {} "$PREFIX/lib/" \;

    # Keep CMake config lookup compatible: oneDALConfig.cmake searches lib/<arch>.
    # Create lib/intel64 links to dal-gpu runtime libs (same layout as release tree).
    mkdir -p "$PREFIX/lib/intel64"
    for lib in "$PREFIX"/lib/libonedal*_dpc*.so*; do
        [ -e "$lib" ] || continue
        libname=$(basename "$lib")
        ln -sf "../$libname" "$PREFIX/lib/intel64/$libname"
    done
fi
if [ "$PKG_NAME" = "dal-static" ]; then
    find lib/intel64 -name "libonedal*.a" -exec cp {} "$PREFIX/lib/" \;
fi
