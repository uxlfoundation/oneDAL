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

OPTFLAGS_SUPPORTED := O0 O1 O2 O3 Os Ofast Og Oz


LINKERS_SUPPORTED := bfd gold lld

ifneq ($(LINKER),)
    ifneq ($(filter $(LINKER),bfd gold lld),$(LINKER))
        $(error Invalid LINKER '$(LINKER)'. Supported on Linux: bfd gold lld)
    endif
endif

ifneq (,$(filter $(OPTFLAG),$(OPTFLAGS_SUPPORTED)))
else
    $(error Invalid OPTFLAG '$(OPTFLAG)' for $(COMPILER). Supported: $(OPTFLAGS_SUPPORTED))
endif

ifeq ($(filter $(OPTFLAG),O0 Og),$(OPTFLAG))
    -optlevel.gnu = -$(OPTFLAG)
else
    -optlevel.gnu = -$(OPTFLAG) -D_FORTIFY_SOURCE=2
endif

linker.ld.flag := $(if $(LINKER),-fuse-ld=$(LINKER),)
link.dynamic.all.gnu = ${CXX} $(linker.ld.flag)

-Zl.gnu = -nostdlib
-DEBC.gnu = -g

-asanstatic.gnu = -static-libasan
-asanshared.gnu =

pedantic.opts.all.gnu = -pedantic \
                        -Wall \
                        -Wextra \
                        -Wno-unused-parameter
