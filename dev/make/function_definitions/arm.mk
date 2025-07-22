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

ifeq ($(filter ref,$(BACKEND_CONFIG)),)
  $(error Unsupported backend config '$(BACKEND_CONFIG)'. \
          Supported config for '$(PLAT)' are ['ref'])
endif

COMPILERs = gnu clang
COMPILER ?= gnu
CPUs := sve
CPUs.files := a8sve
OPTFLAG ?= O2

ONEAPI.dispatcher_tag.a8sve := -D__CPU_TAG__=__CPU_TAG_ARMV8SVE__

# Used as $(eval $(call add_mandatory_cpu,var_name)) to add the mandatory CPU
# sse2 to the start of the list of CPUs stored in 'var_name'
define add_mandatory_cpu
  $$(eval $1 := $$(if $$(filter sve,$$($1)),$$($1),sve $$($1)))
endef

# Used as $(eval $(call set_uarch_options_for_compiler,$(COMPILER)))
define set_uarch_options_for_compiler
  $$(eval a8sve_OPT := $$(a8sve_OPT.$1))
endef

# Used as $(eval $(call set_arch_file_suffix,var_name))
define set_arch_file_suffix
  $$(eval $1.files := $$(subst sve,a8sve,$$($1)))
endef

# Used as $(eval $(call set_usecpu_defs))
# There are no parameters, as we assume we want to update the variable USECPUS,
# but we can't set this without a function call, as we rely on other variables
# already being set
define set_usecpu_defs
  $$(eval USECPUS.out.defs := $$(subst sve,^\#define DAAL_KERNEL_SVE$$(sed.eow),$$(USECPUS.out)))
endef

# Used as $(eval $(call append_uarch_copt,$(OBJNAME)))
define append_uarch_copt
$$(eval $$(call containing,_flt, $1): COPT += -DDAAL_FPTYPE=float)
$$(eval $$(call containing,_dbl, $1): COPT += -DDAAL_FPTYPE=double)
endef

# Used as $(eval $(call subst_arch_cpu_in_var,VARNAME))
define subst_arch_cpu_in_var
  $$(eval $1 := $$(subst _cpu_a8sve,_cpu,$$($1)))
endef

# Use as $(eval $(call add_cpu_to_uarch_in_files,VAR_NAME
define add_cpu_to_uarch_in_files
  $$(eval a8sve_files := $$(subst _a8sve,_cpu_a8sve,$$(call containing,_a8sve,$$($1))))
  $$(eval user_cpu_files := $$(a8sve_files))
endef

# Used as $(eval $(call dispatcher_cpu_rule,rule_name,$(USECPUS))))
define dispatcher_cpu_rule
$1: | $(dir $1)/.
	$(if $(filter sve,$2),echo "#define ONEDAL_CPU_DISPATCH_A8SVE" >> $$@)
endef

# Used as $(eval $(call update_copt_from_dispatcher_tag,$(OBJ_NAME),suffix))
# This must be called after the p4_OPT, mc3_OPT, avx2_OPT, skx_OPT, a8sve_OPT,
# and ONEAPI.dispatcher_tag.* variables are defined. Otherwise this will be a
# no-op
define update_copt_from_dispatcher_tag
  $$(eval $(call containing,_a8sve, $1): COPT += $$(a8sve_OPT$2) $$(ONEAPI.dispatcher_tag.a8sve))
endef
