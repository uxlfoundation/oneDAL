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

export DPL_ROOT=$PREFIX

# Use isolated TBBROOT staging so we don't create files under $PREFIX that may
# accidentally end up in output packages.
# conda-forge tbb-devel may expose only versioned SONAMEs (libtbb.so.<N>,
# libtbbmalloc.so.<N>, etc.) while oneDAL Make expects unversioned names in
# TBBROOT/lib prerequisites.  Mirror the full libtbb* tree into a staging dir,
# adding unversioned symlinks where missing.
export TBBROOT="$SRC_DIR/__tbbroot"
mkdir -p "$TBBROOT/lib" "$TBBROOT/include"

# Symlink TBB headers (TBBROOT/include -> $PREFIX/include)
ln -sfn "$PREFIX/include" "$TBBROOT/include"

# Mirror all libtbb* shared objects and create unversioned symlinks if absent
for versioned in "$PREFIX/lib"/libtbb*.so.*; do
    [ -e "$versioned" ] || continue
    libname=$(basename "$versioned")
    # Link versioned file into staging dir
    ln -sfn "$versioned" "$TBBROOT/lib/$libname"
    # Derive unversioned name: libtbbmalloc.so.2.6 -> libtbbmalloc.so
    unversioned="${libname%%\.so\.*}.so"
    if [ ! -e "$TBBROOT/lib/$unversioned" ]; then
        ln -sfn "$versioned" "$TBBROOT/lib/$unversioned"
    fi
done

# default flags set by conda-build create problems with oneDAL build system
unset CFLAGS LDFLAGS CXXFLAGS
# CONDA_CXX_COMPILER is set by the conda recipe
make -j$(nproc) daal oneapi REQCPU="$REQCPU" COMPILER=$CONDA_CXX_COMPILER
