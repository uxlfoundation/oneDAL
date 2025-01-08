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


-Zl.icx = $(if $(OS_is_win),-Zl,) $(-Q)no-intel-lib
-DEBC.icx = $(if $(OS_is_win),-debug:all -Z7,-g) -fno-system-debug

-Qopt = $(if $(OS_is_win),-Qopt-,-qopt-)

COMPILER.lnx.icx = icx -m64 \
                     -Werror -Wreturn-type -qopenmp-simd
COMPILER.lnx.icx += $(if $(filter yes,$(GCOV_ENABLED)),-coverage,)
COMPILER.win.icx = icx $(if $(MSVC_RT_is_release),-MD -Qopenmp-simd, -MDd) -nologo -WX -Wno-deprecated-declarations

link.dynamic.lnx.icx = icx -m64 -no-intel-lib
link.dynamic.lnx.icx += $(if $(filter yes,$(GCOV_ENABLED)),-coverage,)

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
