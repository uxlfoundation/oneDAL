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
#  g++ definitions for makefile
#  This file contains definitions common to gnu on all platforms. It
#  should only be included from files which have more specializations (e.g.
#  gnu.32e.mk)
#--

CMPLRDIRSUFF.gnu = _gnu

CORE.SERV.COMPILER.gnu = generic

-Zl.gnu =
-DEBC.gnu = -g

-asanstatic.gnu = -static-libasan
-asanshared.gnu =

pedantic.opts.all.gnu = -pedantic \
                        -Wall \
                        -Wextra \
                        -Wno-unused-parameter
