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

BACKEND_CONFIG ?= ref
ARCH = arm
ARCH_DIR_ONEDAL = ARM64
_OS := win
_IA := ARM64

COMPILERs = clang
COMPILER ?= clang

SVE_SUPPORTED := $(shell powershell -NoProfile -Command "Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class Native { [DllImport(\"kernel32.dll\")] public static extern bool IsProcessorFeaturePresent(uint f); }'; if ([Native]::IsProcessorFeaturePresent(46)) { '1' } else { '0' }")

include dev/make/function_definitions/arm.mk

# Used as $(eval $(call set_daal_rt_deps))
define set_daal_rt_deps
  $$(eval daaldep.winarm.rt.thr  := -LIBPATH:$$(RELEASEDIR.tbb.libia) \
          $$(dep_thr) $$(if $$(CHECK_DLL_SIG),Wintrust.lib))
  $$(eval daaldep.winarm.rt.seq  := $$(dep_seq) \
          $$(if $$(CHECK_DLL_SIG),Wintrust.lib))
  $$(eval daaldep.winarm.rt.dpc  := $$(dep_dpc) \
          $$(if $$(CHECK_DLL_SIG),Wintrust.lib))
  $$(eval daaldep.win.threxport.create = grep -v -E '^(;|$$$$$$$$)' $$$$< $$$$(USECPUS.out.grep.filter))
endef

