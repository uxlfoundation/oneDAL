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

ifeq ($(filter mkl ref,$(BACKEND_CONFIG)),)
  $(error Unsupported backend config '$(BACKEND_CONFIG)'. \
          Supported config for '$(PLAT)' are ['mkl', 'ref'])
endif

COMPILERs = icc icx gnu clang vc
COMPILER ?= icx
CPUs := sse2 sse42 avx2 avx512
CPUs.files := nrh neh hsw skx

OPTFLAG ?= O2

ONEAPI.dispatcher_tag.nrh := -D__CPU_TAG__=__CPU_TAG_SSE2__
ONEAPI.dispatcher_tag.neh := -D__CPU_TAG__=__CPU_TAG_SSE42__
ONEAPI.dispatcher_tag.hsw := -D__CPU_TAG__=__CPU_TAG_AVX2__
ONEAPI.dispatcher_tag.skx := -D__CPU_TAG__=__CPU_TAG_AVX512__

# Used as $(eval $(call add_mandatory_cpu,var_name)) to add the mandatory CPU
# sse2 to the start of the list of CPUs stored in 'var_name'
define add_mandatory_cpu
  $$(eval $1 := $$(if $$(filter sse2,$$($1)),$$($1),sse2 $$($1)))
endef

# Used as $(eval $(call set_uarch_options_for_compiler,$(COMPILER)))
define set_uarch_options_for_compiler
  $$(eval p4_OPT := $$(p4_OPT.$1))
  $$(eval mc3_OPT := $$(mc3_OPT.$1))
  $$(eval avx2_OPT := $$(avx2_OPT.$1))
  $$(eval skx_OPT := $$(skx_OPT.$1))
endef

# Used as $(eval $(call set_arch_file_suffix,var_name))
define set_arch_file_suffix
  $$(eval $1.files := $$(subst sse2,nrh,$$(subst sse42,neh,$$(subst avx2,hsw,$$(subst avx512,skx,$$($1))))))
endef

# Used as $(eval $(call set_usecpu_defs))
# There are no parameters, as we assume we want to update the variable USECPUS,
# but we can't set this without a function call, as we rely on other variables
# already being set
define set_usecpu_defs
  $$(eval USECPUS.out.defs := $$(subst sse2,^\#define DAAL_KERNEL_SSE2$$(sed.eow),\
                              $$(subst sse42,^\#define DAAL_KERNEL_SSE42$$(sed.eow),\
                              $$(subst avx2,^\#define DAAL_KERNEL_AVX2$$(sed.eow),\
                              $$(subst avx512,^\#define DAAL_KERNEL_AVX512$$(sed.eow),$$(USECPUS.out))))))
endef

# Used as $(eval $(call append_uarch_copt,$(OBJNAME)))
define append_uarch_copt
$$(eval $$(call containing,_nrh, $1): COPT += $$(p4_OPT)   -DDAAL_CPU=sse2)
$$(eval $$(call containing,_neh, $1): COPT += $$(mc3_OPT)  -DDAAL_CPU=sse42)
$$(eval $$(call containing,_hsw, $1): COPT += $$(avx2_OPT) -DDAAL_CPU=avx2)
$$(eval $$(call containing,_skx, $1): COPT += $$(skx_OPT)  -DDAAL_CPU=avx512)

$$(eval $$(call containing,_flt, $1): COPT += -DDAAL_FPTYPE=float)
$$(eval $$(call containing,_dbl, $1): COPT += -DDAAL_FPTYPE=double)
endef

# Used as $(eval $(call subst_arch_cpu_in_var,VARNAME))
define subst_arch_cpu_in_var
  $$(eval $1 := $$(subst _cpu_nrh,_cpu,$$($1)))
  $$(eval $1 := $$(subst _cpu_neh,_cpu,$$($1)))
  $$(eval $1 := $$(subst _cpu_hsw,_cpu,$$($1)))
  $$(eval $1 := $$(subst _cpu_skx,_cpu,$$($1)))
endef

# Use as $(eval $(call add_cpu_to_uarch_in_files,VAR_NAME
define add_cpu_to_uarch_in_files
  $$(eval nrh_files := $$(subst _nrh,_cpu_nrh,$$(call containing,_nrh,$$($1))))
  $$(eval neh_files := $$(subst _neh,_cpu_neh,$$(call containing,_neh,$$($1))))
  $$(eval hsw_files := $$(subst _hsw,_cpu_hsw,$$(call containing,_hsw,$$($1))))
  $$(eval skx_files := $$(subst _skx,_cpu_skx,$$(call containing,_skx,$$($1))))
  $$(eval user_cpu_files := $$(nrh_files) $$(neh_files) $$(hsw_files) $$(skx_files))
endef

# Used as $(eval $(call dispatcher_cpu_rule,rule_name,$(USECPUS))))
define dispatcher_cpu_rule
$1: | $(dir $1)/.
	$(if $(filter sse42,$2),echo "#define ONEDAL_CPU_DISPATCH_SSE42" >> $$@)
	$(if $(filter avx2,$2),echo "#define ONEDAL_CPU_DISPATCH_AVX2" >> $$@)
	$(if $(filter avx512,$2),echo "#define ONEDAL_CPU_DISPATCH_AVX512" >> $$@)
endef

# Used as $(eval $(call update_copt_from_dispatcher_tag,$(OBJ_NAME),suffix))
# This must be called after the p4_OPT, mc3_OPT, avx2_OPT, skx_OPT, a8sve_OPT,
# and ONEAPI.dispatcher_tag.* variables are defined. Otherwise this will be a
# no-op
define update_copt_from_dispatcher_tag
  $$(eval $(call containing,_nrh, $1): COPT += $$(p4_OPT$2)   $$(ONEAPI.dispatcher_tag.nrh))
  $$(eval $(call containing,_neh, $1): COPT += $$(mc3_OPT$2)  $$(ONEAPI.dispatcher_tag.neh))
  $$(eval $(call containing,_hsw, $1): COPT += $$(avx2_OPT$2) $$(ONEAPI.dispatcher_tag.hsw))
  $$(eval $(call containing,_skx, $1): COPT += $$(skx_OPT$2)  $$(ONEAPI.dispatcher_tag.skx))
endef
