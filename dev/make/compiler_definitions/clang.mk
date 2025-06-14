# file: clang.mk
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

#++
#  Clang definitions for makefile
#  This file contains definitions common to clang on all platforms.
#  It should only be included from files which have more specializations (e.g.
#  clang.32e.mk)
#--

CMPLRDIRSUFF.clang = _clang

CORE.SERV.COMPILER.clang = generic

-Zl.clang =
-DEBC.clang = -g

-asanstatic.clang = -static-libasan
-asanshared.clang = -shared-libasan

pedantic.opts.clang = -pedantic \
                      -Wall \
                      -Wextra \
                      -Wno-unused-parameter
