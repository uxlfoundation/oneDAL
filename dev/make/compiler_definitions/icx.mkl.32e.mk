#===============================================================================
# Copyright 2022 Intel Corporation
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
#  Intel compiler definitions for makefile
#--

PLATs.icx = lnx32e win32e

CMPLRDIRSUFF.icx =

CORE.SERV.COMPILER.icx = generic

OPTFLAGS_SUPPORTED := O0 O1 O2 O3 Ofast Os Oz Og

LINKERS_SUPPORTED := bfd gold lld llvm-lib

ifeq ($(OS_is_win),true)
    ifneq ($(LINKER),)
        ifneq ($(filter $(LINKER),lld llvm-lib),$(LINKER))
            $(error Invalid LINKER '$(LINKER)'. Supported on Windows: lld llvm-lib)
        endif
    endif
else
    ifneq ($(LINKER),)
        ifneq ($(filter $(LINKER),bfd gold lld),$(LINKER))
            $(error Invalid LINKER '$(LINKER)'. Supported on Linux: bfd gold lld)
        endif
    endif
endif

ifneq (,$(filter $(OPTFLAG),$(OPTFLAGS_SUPPORTED)))
else
    $(error Invalid OPTFLAG '$(OPTFLAG)' for $(COMPILER). Supported: $(OPTFLAGS_SUPPORTED))
endif

ifeq ($(OS_is_win),true)
    -optlevel.icx = -$(OPTFLAG)
else
    ifeq ($(OPTFLAG),Ofast)
        -optlevel.icx = -O3 -ffast-math -D_FORTIFY_SOURCE=2
    else ifeq ($(OPTFLAG),O0)
        -optlevel.icx = -$(OPTFLAG)
    else
        -optlevel.icx = -$(OPTFLAG) -D_FORTIFY_SOURCE=2
    endif
endif

-Zl.icx = $(if $(OS_is_win),-Zl,)
-DEBC.icx = $(if $(OS_is_win),-debug:all -Z7,-g) -fno-system-debug

-asanstatic.icx = -static-libasan
-asanshared.icx = -shared-libasan

-Qopt = $(if $(OS_is_win),-Qopt-,-qopt-)

COMPILER.lnx.icx = icx -m64 -no-intel-lib \
                     -Werror -Wreturn-type -qopenmp-simd ${CXXFLAGS}
COMPILER.lnx.icx += $(if $(filter yes,$(GCOV_ENABLED)),-coverage,)
COMPILER.win.icx = icx $(if $(MSVC_RT_is_release),-MD -Qopenmp-simd, -MDd) -Qno-intel-lib -nologo -WX -Wno-deprecated-declarations ${CXXFLAGS}

linker.ld.flag := $(if $(LINKER),-fuse-ld=$(LINKER),)

link.dynamic.lnx.icx = icx $(linker.ld.flag) -m64 -no-intel-lib ${LDFLAGS}
link.dynamic.lnx.icx += $(if $(filter yes,$(GCOV_ENABLED)),-coverage,)
link.dynamic.win.icc = icx $(linker.ld.flag) ${LDFLAGS}

pedantic.opts.lnx.icx = -pedantic \
                        -Wall \
                        -Wextra \
                        -Wwritable-strings \
                        -Wno-unused-parameter

pedantic.opts.win.icx = -Wall \
                        -Wextra \
                        -Wwritable-strings \
                        -Wno-unused-parameter

p4_OPT.icx   = -march=nocona
mc3_OPT.icx  = -march=nehalem
avx2_OPT.icx = -march=haswell
skx_OPT.icx  = -march=skx
