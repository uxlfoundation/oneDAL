# file: clang.32e.mk
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
#  Clang definitions for makefile.
#  This file contains definitions common to clang on a 32e (intel64) platform.
#  It should only be included from files which have more specializations (e.g.
#  clang.mkl.32e.mk)
#--

include dev/make/compiler_definitions/clang.mk

PLATs.clang = lnx32e mac32e

OPTFLAGS_SUPPORTED := O0 O1 O2 O3 Ofast Os Oz Og

ifneq (,$(filter $(OPTFLAG),$(OPTFLAGS_SUPPORTED)))
else
    $(error Invalid OPTFLAG '$(OPTFLAG)' for $(COMPILER). Supported: $(OPTFLAGS_SUPPORTED))
endif


ifeq ($(filter $(OPTFLAG),O0 Og),$(OPTFLAG))
    -optlevel.clang = -$(OPTFLAG)
else
    -optlevel.clang = -$(OPTFLAG) -D_FORTIFY_SOURCE=2
endif

COMPILER.mac.clang = clang++ -m64 -fgnu-runtime -stdlib=libc++ -mmacosx-version-min=10.15 -fwrapv \
                     -Werror -Wreturn-type
COMPILER.lnx.clang = clang++ -m64 \
                     -Werror -Wreturn-type

link.dynamic.mac.clang = clang++ -m64
link.dynamic.lnx.clang = clang++ -m64

pedantic.opts.mac.clang = $(pedantic.opts.clang)
pedantic.opts.lnx.clang = $(pedantic.opts.clang)

p4_OPT.clang   = $(-Q)march=nocona
mc3_OPT.clang  = $(-Q)$(if $(OS_is_mac),march=nocona,march=nehalem) $(if $(OS_is_mac),$(-Q)mtune=nehalem)
avx2_OPT.clang = $(-Q)march=haswell
skx_OPT.clang  = $(-Q)march=skx
