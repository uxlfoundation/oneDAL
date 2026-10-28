# file: cmplt.dpcpp.mk
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
#  DPC++ Compiler definitions for makefile
#--

PLATs.dpcpp = lnx32e win32e

CMPLRDIRSUFF.dpcpp = _dpcpp

CORE.SERV.COMPILER.dpcpp = generic

OPTFLAGS_SUPPORTED := O0 O1 O2 O3 Os Oz Og

ifneq (,$(filter $(OPTFLAG),$(OPTFLAGS_SUPPORTED)))
else
    $(error Invalid OPTFLAG '$(OPTFLAG)' for $(COMPILER). Supported: $(OPTFLAGS_SUPPORTED))
endif

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

ifeq ($(OS_is_win),true)
    ifeq ($(filter $(OPTFLAG),Ob Od Oi Os Ot Ox Oy),$(OPTFLAG))
        -optlevel.dpcpp = -O2
    else
        -optlevel.dpcpp = -$(OPTFLAG)
    endif
else
    # -U first: a different level from CXXFLAGS would otherwise be a -Werror redefinition.
    ifeq ($(OPTFLAG),O0)
        -optlevel.dpcpp = -$(OPTFLAG)
    else
        -optlevel.dpcpp = -$(OPTFLAG) -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2
    endif
endif


-Zl.dpcpp = $(if $(OS_is_win),-Zl -Q,-)no-intel-lib
-DEBC.dpcpp = $(if $(OS_is_win),-debug:all -Z7,-g) -fno-system-debug

-asanstatic.dpcpp = -static-libasan
-asanshared.dpcpp = -shared-libasan

COMPILER.lnx.dpcpp = icpx -fsycl -m64 -stdlib=libstdc++ -fgnu-runtime -fwrapv \
                     -Werror -Wreturn-type -fsycl-device-code-split=per_kernel
COMPILER.win.dpcpp = icx -fsycl $(if $(MSVC_RT_is_release),-MD, -MDd /debug:none) -nologo -WX \
                     -Wno-deprecated-declarations -Wno-ignored-attributes -fsycl-device-code-split=per_kernel

link.dynamic.lnx.dpcpp = icpx $(linker.ld.flag) -fsycl -m64 -lgomp \
                     -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=$(SYCL_LINK_PRL)
link.dynamic.lnx.dpcpp += $(if $(filter yes,$(GCOV_ENABLED)),-Xscoverage,)

# Device link with the new offloading driver, as in the Bazel toolchain
# (dev/bazel/toolchains/cc_toolchain_lnx.bzl). The legacy driver unbundles every
# input object, including the host-only DAAL objects; the new one skips them.
# Link actions only; compile actions and object formats are unchanged. The flags
# are set per DPC library target in makefile, not here, because LOPT is the link
# options variable and this one is global. Empty if icpx does not support the
# option.
offload.new.driver.lnx.dpcpp := $(if $(OS_is_lnx),$(shell icpx -fsycl --offload-new-driver -fsyntax-only -x c++ /dev/null >/dev/null 2>&1 && echo yes))

link.dynamic.win.dpcpp = icx $(linker.ld.flag) -fsycl -m64 \
                     -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=$(SYCL_LINK_PRL)

pedantic.opts.lnx.dpcpp = -pedantic \
                          -Wall \
                          -Wextra \
                          -Wwritable-strings \
                          -Wno-unused-parameter

pedantic.opts.win.dpcpp = -Wall \
                          -Wextra \
                          -Wwritable-strings \
                          -Wno-unused-parameter

p4_OPT.dpcpp   = -march=nocona
mc3_OPT.dpcpp  = -march=nehalem
avx2_OPT.dpcpp = -march=haswell
skx_OPT.dpcpp  = -march=skx
