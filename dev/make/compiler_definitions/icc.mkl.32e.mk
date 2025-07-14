#===============================================================================
# Copyright 2012 Intel Corporation
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

PLATs.icc = lnx32e win32e mac32e

CMPLRDIRSUFF.icc =

CORE.SERV.COMPILER.icc = generic
-DEBC.icc = $(if $(OS_is_win),-debug:all -Z7,-g)

OPTFLAGS_SUPPORTED := O0 O1 O2 O3 Ofast Os Oz Og

ifneq (,$(filter $(OPTFLAG),$(OPTFLAGS_SUPPORTED)))
else
    $(error Invalid OPTFLAG '$(OPTFLAG)' for $(COMPILER). Supported: $(OPTFLAGS_SUPPORTED))
endif

ifeq ($(OS_is_win),true)
    -optlevel.icx = -$(OPTFLAG)
else
    ifeq ($(OPTFLAG),Ofast)
        -optlevel.icc = -O3 -ffast-math
    else ifeq ($(OPTFLAG),O0)
        -optlevel.icc = -$(OPTFLAG)
    else
        -optlevel.icc = -$(OPTFLAG) -D_FORTIFY_SOURCE=2
    endif
endif

-Zl.icc = $(if $(OS_is_win),-Zl,) -mGLOB_freestanding=TRUE -mCG_no_libirc=TRUE
-Qopt = $(if $(OS_is_win),-Qopt-,-qopt-)

COMPILER.lnx.icc  = $(if $(COVFILE),cov01 -1; covc --no-banner -i )icc -qopenmp-simd \
                    -Werror -Wreturn-type -diag-disable=10441
COMPILER.lnx.icc += $(if $(COVFILE), $(-Q)m64)
COMPILER.win.icc = icl $(if $(MSVC_RT_is_release),-MD, -MDd /debug:none) -nologo -WX -Qopenmp-simd -Qdiag-disable:10441
COMPILER.mac.icc = icc -stdlib=libc++ -mmacosx-version-min=10.15 \
				   -Werror -Wreturn-type -diag-disable=10441

link.dynamic.lnx.icc = icc -no-cilk -diag-disable=10441
link.dynamic.mac.icc = icc -diag-disable=10441

pedantic.opts.lnx.icc = -pedantic \
                        -Wall \
                        -Wextra \
                        -Wno-unused-parameter

daaldep.lnx32e.rt.icc = -static-intel
daaldep.lnx32.rt.icc  = -static-intel

p4_OPT.icc   = $(-Q)$(if $(OS_is_mac),march=pentium4,xSSE2)
mc3_OPT.icc  = $(-Q)xSSE4.2
avx2_OPT.icc = $(-Q)xCORE-AVX2
skx_OPT.icc  = $(-Q)xCORE-AVX512 $(-Qopt)zmm-usage=high
#TODO add march opts in GCC style
