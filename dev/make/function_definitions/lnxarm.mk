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
ARCH_DIR_ONEDAL = arm
_OS := lnx
_IA := arm

include dev/make/function_definitions/arm.mk

# Used as $(eval $(call set_daal_rt_deps))
define set_daal_rt_deps
  $$(eval daaldep.lnxarm.rt.thr := -L$$(TBBDIR.soia.lnx) -ltbb -ltbbmalloc \
          -lpthread $$(daaldep.lnxarm.rt.$$(COMPILER)) \
          $$(if $$(COV.libia),$$(COV.libia)/libcov.a))
  $$(eval daaldep.lnxarm.rt.seq := -lpthread $$(daaldep.lnxarm.rt.$$(COMPILER)) \
	  	  $$(if $$(RNG_OPENRNG), $$(daaldep.rng_backend.lib)) \
          $$(if $$(COV.libia),$$(COV.libia)/libcov.a))
  $$(eval daaldep.lnxarm.rt.dpc := -lpthread -lOpenCL \
          $$(if $$(COV.libia),$$(COV.libia)/libcov.a))
  $$(eval daaldep.lnxarm.threxport := export_lnxarm.$$(BACKEND_CONFIG).def)

  $$(eval daaldep.lnx.threxport.create = grep -v -E '^(EXPORTS|;|$$$$$$$$)' $$$$< $$$$(USECPUS.out.grep.filter) | sed -e 's/^/-u /')
endef
