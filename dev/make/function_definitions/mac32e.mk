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

BACKEND_CONFIG ?= mkl
ARCH = 32e
ARCH_DIR_ONEDAL = intel64
_OS := mac
_IA := intel64

include dev/make/function_definitions/32e.mk

# Used as $(eval $(call set_daal_rt_deps))
define set_daal_rt_deps
  $$(eval daaldep.mac32e.rt.thr := -L$$(RELEASEDIR.tbb.soia) -ltbb -ltbbmalloc \
          $$(daaldep.mac32e.rt.$$(COMPILER)))
  $$(eval daaldep.mac32e.rt.seq := $$(daaldep.mac32e.rt.$$(COMPILER)))

  $$(eval daaldep.mac.threxport.create = grep -v -E '^(EXPORTS|;|$$$$$$$$)' $$$$< $$$$(USECPUS.out.grep.filter) | sed -e 's/^/-u /')
endef
