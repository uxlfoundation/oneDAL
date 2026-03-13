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
ARCH = riscv64
ARCH_DIR_ONEDAL = riscv64
_OS := lnx
_IA := riscv64

include dev/make/function_definitions/riscv64.mk

# Used as $(eval $(call set_daal_rt_deps))
define set_daal_rt_deps
  $$(eval daaldep.lnxriscv64.rt.thr := -L$$(TBBDIR.soia.lnx) -ltbb -ltbbmalloc \
          -lpthread $$(daaldep.lnxriscv64.rt.$$(COMPILER)) \
          $$(if $$(COV.libia),$$(COV.libia)/libcov.a))
  $$(eval daaldep.lnxriscv64.rt.seq := -lpthread $$(daaldep.lnxriscv64.rt.$$(COMPILER)) \
          $$(if $$(COV.libia),$$(COV.libia)/libcov.a))
  $$(eval daaldep.lnxriscv64.rt.dpc := -lpthread \
          $$(if $$(COV.libia),$$(COV.libia)/libcov.a))
  $$(eval daaldep.lnxriscv64.threxport := export_lnxriscv64.$$(BACKEND_CONFIG).def)

  $$(eval daaldep.lnx.threxport.create = grep -v -E '^(EXPORTS|;|$$$$$$$$)' $$$$< $$$$(USECPUS.out.grep.filter) | sed -e 's/^/-u /')
endef
