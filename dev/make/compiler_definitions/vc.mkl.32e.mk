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
#  Visual Studio definitions for makefile
#--

PLATs.vc = win32e

CMPLRDIRSUFF.vc = _vc

CORE.SERV.COMPILER.vc = generic

-Zl.vc = -Zl
-DEBC.vc = -DEBUG -Z7

# Disable C4661 because of false positives
COMPILER.win.vc = cl $(if $(MSVC_RT_is_release),-MD, -MDd) -nologo -EHsc -wd4661 -WX

link.dynamic.win.vc = /DEPENDENTLOADFLAG:0x2000

p4_OPT.vc   =
mc3_OPT.vc  =
avx2_OPT.vc = -arch:AVX2
skx_OPT.vc  = -arch:AVX2
