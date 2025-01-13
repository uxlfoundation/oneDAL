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

-Zl.dpcpp = $(if $(OS_is_win),-Zl -Q,-)no-intel-lib
-DEBC.dpcpp = $(if $(OS_is_win),-debug:all -Z7,-g) -fno-system-debug

COMPILER.lnx.dpcpp = icpx -fsycl -m64 -stdlib=libstdc++ -fgnu-runtime -fwrapv \
                     -Werror -Wreturn-type -fsycl-device-code-split=per_kernel
COMPILER.win.dpcpp = icx -fsycl $(if $(MSVC_RT_is_release),-MD, -MDd /debug:none) -nologo -WX \
                     -Wno-deprecated-declarations -fsycl-device-code-split=per_kernel

link.dynamic.lnx.dpcpp = icpx -fsycl -m64 -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=$(SYCL_LINK_PRL)
link.dynamic.lnx.dpcpp += $(if $(filter yes,$(GCOV_ENABLED)),-Xscoverage,)

link.dynamic.win.dpcpp = icx -fsycl -m64 -fsycl-device-code-split=per_kernel -fsycl-max-parallel-link-jobs=$(SYCL_LINK_PRL)

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
