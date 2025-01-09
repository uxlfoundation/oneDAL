#===============================================================================
# Copyright 2014 Intel Corporation
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

#===============================================================================
# Common macros
#===============================================================================

IDENTIFIED_PLAT=$(shell bash dev/make/identify_os.sh)
ifeq (help,$(MAKECMDGOALS))
    PLAT := win32e
else ifeq ($(PLAT),)
    PLAT := $(IDENTIFIED_PLAT)
endif

# Check that we know how to build for the identified platform
PLATs := lnx32e mac32e win32e lnxarm lnxriscv64
$(if $(filter $(PLAT),$(PLATs)),,$(error Unknown platform $(PLAT)))

# Non-platform or architecture specific defines live in common.mk
include dev/make/common.mk

# Platform specific variables are set in dev/make/function_definitions/$(PLAT).mk
# There are also files dev/make/function_definitions/$(ARCH).mk, but these are included from
# the $(PLAT).mk files, rather than here.
include dev/make/function_definitions/$(PLAT).mk

$(if $(filter $(COMPILERs),$(COMPILER)),,$(error COMPILER must be one of $(COMPILERs)))

MSVC_RUNTIME_VERSIONs = release debug
MSVC_RUNTIME_VERSION ?= release
$(if $(filter $(MSVC_RUNTIME_VERSIONs),$(MSVC_RUNTIME_VERSION)),,$(error MSVC_RUNTIME_VERSION must be one of $(MSVC_RUNTIME_VERSIONs)))

MAKE_PID := $(shell echo $$PPID)
JOB_FLAG := $(filter -j%, $(subst -j ,-j,$(shell ps T | grep "^\s*$(MAKE_PID).*$(MAKE)")))
MAKE_JOBS := $(subst -j,,$(JOB_FLAG))
MAKE_JOBS := $(if $(filter $(MAKE_JOBS),$(shell seq 1 999)),$(MAKE_JOBS),$(shell nproc))

SYCL_LINK_PRL := $(MAKE_JOBS)

COMPILER_is_$(COMPILER)            := yes
COMPILER_is_cross                  := $(if $(filter $(PLAT),$(IDENTIFIED_PLAT)),no,yes)
OS_is_$(_OS)                       := yes
IA_is_$(_IA)                       := yes
PLAT_is_$(PLAT)                    := yes
MSVC_RT_is_$(MSVC_RUNTIME_VERSION) := yes
ARCH_is_$(ARCH)                    := yes

GCOV_ENABLED                       := $(if $(filter yes,$(CODE_COVERAGE)),$(if $(OS_is_lnx),yes,no),no)

DEFAULT_BUILD_PARAMETERS_LIB       := $(if $(OS_is_win),no,yes)
BUILD_PARAMETERS_LIB               ?= $(DEFAULT_BUILD_PARAMETERS_LIB)

ifdef OS_is_win
ifeq ($(BUILD_PARAMETERS_LIB),yes)
$(error Building with the parameters library is not available on Windows OS)
endif
endif

USERREQCPU := $(filter-out $(filter $(CPUs),$(REQCPU)),$(REQCPU))
USECPUS := $(if $(REQCPU),$(if $(USERREQCPU),$(error Unsupported value/s in REQCPU: $(USERREQCPU). List of supported CPUs: $(CPUs)),$(REQCPU)),$(CPUs))

$(eval $(call add_mandatory_cpu,USECPUS))

$(info Selected list of CPUs - USECPUS: $(USECPUS))

req-features = order-only second-expansion
ifneq ($(words $(req-features)),$(words $(filter $(req-features),$(.FEATURES))))
$(error This makefile requires a decent make, supporting $(req-features))
endif

.PHONY: help
help: ; $(info $(help))

#===============================================================================
# Compiler specific part
#===============================================================================

ifeq ($(OS_is_lnx),yes)
GCC_TOOLCHAIN_PATH := $(realpath $(dir $(shell which gcc))/..)

ifeq ($(COMPILER_is_clang),yes)
C.COMPILE.gcc_toolchain := $(GCC_TOOLCHAIN_PATH)
endif

DPC.COMPILE.gcc_toolchain := $(GCC_TOOLCHAIN_PATH)
endif

include dev/make/compiler_definitions/$(COMPILER).$(BACKEND_CONFIG).$(ARCH).mk
include dev/make/compiler_definitions/dpcpp.mk

$(if $(filter $(PLATs.$(COMPILER)),$(PLAT)),,$(error PLAT for $(COMPILER) must be defined to one of $(PLATs.$(COMPILER))))

#===============================================================================
# Dependencies generation
#===============================================================================

include dev/make/deps.mk

#===============================================================================
# Common macros
#===============================================================================

AR_is_$(subst $(space),_,$(origin AR)) := yes

OSList          := lnx win mac

o           := $(if $(OS_is_win),obj,o)
a           := $(if $(OS_is_win),lib,a)
d           := $(if $(OS_is_win),$(if $(MSVC_RT_is_debug),d,),)
dtbb        := $(if $(OS_is_win),$(if $(MSVC_RT_is_debug),_debug,),)
plib        := $(if $(OS_is_win),,lib)
scr         := $(if $(OS_is_win),bat,sh)
y           := $(notdir $(filter $(_OS)/%,lnx/so win/dll mac/dylib))
-Fo         := $(if $(OS_is_win),-Fo,-o)
-Q          := $(if $(OS_is_win),$(if $(COMPILER_is_vc),-,-Q),-)
-cxx17      := $(if $(COMPILER_is_vc),/std:c++17,$(-Q)std=c++17)
-fPIC       := $(if $(OS_is_win),,-fPIC)
-DMKL_ILP64 := $(if $(filter mkl,$(BACKEND_CONFIG)),-DMKL_ILP64)
-Zl         := $(-Zl.$(COMPILER))
-Zl_DPCPP   := $(-Zl.dpcpp)
-DEBC       := $(if $(REQDBG),$(-DEBC.$(COMPILER)) -DDEBUG_ASSERT -DONEDAL_ENABLE_ASSERT) -DTBB_SUPPRESS_DEPRECATED_MESSAGES -D__TBB_LEGACY_MODE
-DEBC_DPCPP := $(if $(REQDBG),$(-DEBC.dpcpp) -DDEBUG_ASSERT -DONEDAL_ENABLE_ASSERT)
-DEBL       := $(if $(REQDBG),$(if $(OS_is_win),-debug,))
-EHsc       := $(if $(OS_is_win),-EHsc,)
-isystem    := $(if $(OS_is_win),-I,-isystem)
-sGRP       := $(if $(OS_is_lnx),-Wl$(comma)--start-group,)
-eGRP       := $(if $(OS_is_lnx),-Wl$(comma)--end-group,)
daalmake = make

$(eval $(call set_uarch_options_for_compiler,$(COMPILER)))

$(eval $(call set_arch_file_suffix,USECPUS))

USECPUS.out := $(filter-out $(USECPUS),$(CPUs))
USECPUS.out.for.grep.filter := $(addprefix _,$(addsuffix _,$(subst $(space),_|_,$(USECPUS.out))))
USECPUS.out.grep.filter := $(if $(USECPUS.out),| grep -v -E '$(USECPUS.out.for.grep.filter)')

$(eval $(call set_usecpu_defs))

USECPUS.out.defs := $(subst $(space)^,|^,$(strip $(USECPUS.out.defs)))
USECPUS.out.defs.filter := $(if $(USECPUS.out.defs),sed $(sed.-b) $(sed.-i) -E -e 's/$(USECPUS.out.defs)/$(sed.eol)/')

#===============================================================================
# Paths
#===============================================================================

# LINUX release structure (under __release_lnx):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries
# daal/lib/intel64 - static and dynamic libraries for intel64

# macOS* release structure (under __release_mac):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries, and Mach-O intel64 binaries

# WINDOWS release structure (under __release_win):
# daal
# daal/bin - platform independent binaries: env setters
# daal/examples - usage demonstrations
# daal/include - header files
# daal/lib - platform-independent libraries
# daal/lib/intel64 - static and import libraries for intel64
# redist/intel64/daal - dlls for intel64

# List of needed threadings layers can be specified in DAALTHRS.
# if DAALTHRS is empty, threading will be incapsulated to core
DAALTHRS ?= tbb
DAALAY   ?= a y

DIR:=.
CPPDIR:=$(DIR)/cpp
CPPDIR.daal:=$(CPPDIR)/daal
CPPDIR.onedal:=$(CPPDIR)/oneapi/dal
WORKDIR    ?= $(DIR)/__work$(CMPLRDIRSUFF.$(COMPILER))/$(if $(MSVC_RT_is_release),md,mdd)/$(PLAT)
RELEASEDIR ?= $(DIR)/__release_$(_OS)$(CMPLRDIRSUFF.$(COMPILER))
RELEASEDIR.daal        := $(RELEASEDIR)/daal/latest
RELEASEDIR.lib         := $(RELEASEDIR.daal)/lib
RELEASEDIR.env         := $(RELEASEDIR.daal)/env
RELEASEDIR.modulefiles := $(RELEASEDIR.daal)/modulefiles
RELEASEDIR.conf        := $(RELEASEDIR.daal)/config
RELEASEDIR.nuspec      := $(RELEASEDIR.daal)/nuspec
RELEASEDIR.doc         := $(RELEASEDIR.daal)/documentation
RELEASEDIR.samples     := $(RELEASEDIR.daal)/samples
RELEASEDIR.libia       := $(RELEASEDIR.daal)/lib$(if $(OS_is_mac),,/$(_IA))
RELEASEDIR.include     := $(RELEASEDIR.daal)/include
RELEASEDIR.pkgconfig   := $(RELEASEDIR.daal)/lib/pkgconfig
RELEASEDIR.soia        := $(if $(OS_is_win),$(RELEASEDIR.daal)/redist/$(_IA),$(RELEASEDIR.libia))
WORKDIR.lib := $(WORKDIR)/daal/lib

COVFILE   := $(subst BullseyeStub,$(RELEASEDIR.daal)/Bullseye_$(_IA).cov,$(COVFILE))
COV.libia := $(if $(BULLSEYEROOT),$(BULLSEYEROOT)/lib)

topf = $(shell echo $1 | sed 's/ /111/g' | sed 's/(/222/g' | sed 's/)/333/g' | sed 's/\\/\//g')
frompf = $(shell echo $1 | sed 's/111/ /g' | sed 's/222/(/g' | sed 's/333/)/g')
frompf1 = $(shell echo $1 | sed 's/111/\\ /g' | sed 's/222/(/g' | sed 's/333/)/g')


#============================= TBB folders =====================================
TBBDIR := $(if $(wildcard $(DIR)/__deps/tbb/$(_OS)/*),$(DIR)/__deps/tbb/$(_OS)$(if $(OS_is_win),/tbb))
TBBDIR.2 := $(if $(TBBDIR),$(TBBDIR),$(call topf,$$TBBROOT))
TBBDIR.2 := $(if $(TBBDIR.2),$(TBBDIR.2),$(error Can`t find TBB neither in $(DIR)/__deps/tbb not in $$TBBROOT))

TBBDIR.include := $(if $(TBBDIR.2),$(TBBDIR.2)/include/tbb $(TBBDIR.2)/include)

TBBDIR.libia.prefix := $(TBBDIR.2)/lib

OLD_TBB_LAYOUT := $(if $(wildcard $(call frompf1,$(TBBDIR.libia.prefix))/$(_IA)),yes,no)

ifeq ($(OLD_TBB_LAYOUT),no)
  TBBDIR.libia.win.vc1  := $(if $(OS_is_win),$(if $(wildcard $(call frompf1,$(TBBDIR.libia.prefix))/vc_mt),$(TBBDIR.libia.prefix)/vc_mt,$(if $(wildcard $(call frompf1,$(TBBDIR.libia.prefix))/vc14),$(TBBDIR.libia.prefix)/vc14)))
else
  TBBDIR.libia.win.vc1  := $(if $(OS_is_win),$(if $(wildcard $(call frompf1,$(TBBDIR.libia.prefix))/$(_IA)/vc_mt),$(TBBDIR.libia.prefix)/$(_IA)/vc_mt,$(if $(wildcard $(call frompf1,$(TBBDIR.libia.prefix))/$(_IA)/vc14),$(TBBDIR.libia.prefix)/$(_IA)/vc14)))
endif
TBBDIR.libia.win.vc2  := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc1),,$(firstword $(filter $(call topf,$$TBBROOT)%,$(subst ;,$(space),$(call topf,$$LIB))))))
TBBDIR.libia.win.vc22 := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc2),$(wildcard $(TBBDIR.libia.win.vc2)/tbb12$(dtbb).dll)))

ifeq ($(OLD_TBB_LAYOUT),no)
  TBBDIR.libia.win:= $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(TBBDIR.libia.win.vc1),$(TBBDIR.libia.win.vc1),$(error Can`t find TBB libs nether in $(call frompf,$(TBBDIR.libia.prefix))/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).))))
  TBBDIR.libia.lnx.gcc1 := $(if $(OS_is_lnx),$(if $(wildcard $(TBBDIR.libia.prefix)/*),$(TBBDIR.libia.prefix)))
else
  TBBDIR.libia.win:= $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(TBBDIR.libia.win.vc1),$(TBBDIR.libia.win.vc1),$(error Can`t find TBB libs nether in $(call frompf,$(TBBDIR.libia.prefix))/$(_IA)/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).))))
  TBBDIR.libia.lnx.gcc1 := $(if $(OS_is_lnx),$(if $(wildcard $(TBBDIR.libia.prefix)/$(_IA)/gcc4.8/*),$(TBBDIR.libia.prefix)/$(_IA)/gcc4.8))
endif

TBBDIR.libia.lnx.gcc2  := $(if $(OS_is_lnx),$(if $(TBBDIR.libia.lnx.gcc1),,$(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LD_LIBRARY_PATH))))))
TBBDIR.libia.lnx.gcc22 := $(if $(OS_is_lnx),$(if $(TBBDIR.libia.lnx.gcc2),$(wildcard $(TBBDIR.libia.lnx.gcc2)/libtbb.so)))
ifeq ($(OLD_TBB_LAYOUT),no)
  TBBDIR.libia.lnx := $(if $(OS_is_lnx),$(if $(TBBDIR.libia.lnx.gcc22),$(TBBDIR.libia.lnx.gcc2),$(if $(TBBDIR.libia.lnx.gcc1),$(TBBDIR.libia.lnx.gcc1),$(error Can`t find TBB runtimes nether in $(TBBDIR.libia.prefix) not in $(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LD_LIBRARY_PATH)))).))))
else
  TBBDIR.libia.lnx := $(if $(OS_is_lnx),$(if $(TBBDIR.libia.lnx.gcc22),$(TBBDIR.libia.lnx.gcc2),$(if $(TBBDIR.libia.lnx.gcc1),$(TBBDIR.libia.lnx.gcc1),$(error Can`t find TBB runtimes nether in $(TBBDIR.libia.prefix)/$(_IA)/gcc4.8 not in $(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LD_LIBRARY_PATH)))).))))
endif
TBBDIR.libia.mac.clang1  := $(if $(OS_is_mac),$(if $(wildcard $(TBBDIR.libia.prefix)/*),$(TBBDIR.libia.prefix)))
TBBDIR.libia.mac.clang2  := $(if $(OS_is_mac),$(if $(TBBDIR.libia.mac.clang1),,$(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LIBRARY_PATH))))))
TBBDIR.libia.mac.clang22 := $(if $(OS_is_mac),$(if $(TBBDIR.libia.mac.clang2),$(wildcard $(TBBDIR.libia.mac.clang2)/libtbb.dylib)))
TBBDIR.libia.mac := $(if $(OS_is_mac),$(if $(TBBDIR.libia.mac.clang22),$(TBBDIR.libia.mac.clang2),$(if $(TBBDIR.libia.mac.clang1),$(TBBDIR.libia.mac.clang1),$(error Can`t find TBB runtimes nether in $(TBBDIR.libia.prefix) not in $(firstword $(filter $(TBBROOT)%,$(subst :,$(space),$(LIBRARY_PATH)))).))))

TBBDIR.libia := $(TBBDIR.libia.$(_OS))

TBBDIR.soia.prefix := $(TBBDIR.2)/
ifeq ($(OLD_TBB_LAYOUT),no)
  TBBDIR.soia.win  := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(wildcard $(call frompf1,$(TBBDIR.soia.prefix))bin/vc_mt/*),$(TBBDIR.soia.prefix)bin/vc_mt,$(if $(wildcard $(call frompf1,$(TBBDIR.soia.prefix))bin/vc14/*),$(TBBDIR.soia.prefix)bin/vc14,$(error Can`t find TBB runtimes nether in $(TBBDIR.soia.prefix)bin/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).)))))
else
  TBBDIR.soia.win  := $(if $(OS_is_win),$(if $(TBBDIR.libia.win.vc22),$(TBBDIR.libia.win.vc2),$(if $(wildcard $(call frompf1,$(TBBDIR.soia.prefix))redist/$(_IA)/vc_mt/*),$(TBBDIR.soia.prefix)redist/$(_IA)/vc_mt,$(if $(wildcard $(call frompf1,$(TBBDIR.soia.prefix))redist/$(_IA)/vc14/*),$(TBBDIR.soia.prefix)redist/$(_IA)/vc14,$(error Can`t find TBB runtimes nether in $(TBBDIR.soia.prefix)redist/$(_IA)/vc_mt not in $(firstword $(filter $(TBBROOT)%,$(subst ;,$(space),$(LIB)))).)))))
endif
TBBDIR.soia.lnx  := $(if $(OS_is_lnx),$(TBBDIR.libia.lnx))
TBBDIR.soia.mac  := $(if $(OS_is_mac),$(TBBDIR.libia.mac))
TBBDIR.soia := $(TBBDIR.soia.$(_OS))

RELEASEDIR.tbb       := $(RELEASEDIR)/tbb/latest
ifeq ($(OLD_TBB_LAYOUT),no)
  RELEASEDIR.tbb.libia := $(RELEASEDIR.tbb)/lib$(if $(OS_is_mac),,$(if $(OS_is_win),/vc_mt,/$(TBBDIR.libia.lnx.gcc)))
  RELEASEDIR.tbb.soia  := $(if $(OS_is_win),$(RELEASEDIR.tbb)/bin/vc_mt,$(RELEASEDIR.tbb.libia))
else
  RELEASEDIR.tbb.libia := $(RELEASEDIR.tbb)/lib$(if $(OS_is_mac),,/$(_IA)$(if $(OS_is_win),/vc_mt,/$(TBBDIR.libia.lnx.gcc)))
  RELEASEDIR.tbb.soia  := $(if $(OS_is_win),$(RELEASEDIR.tbb)/redist/$(_IA)/vc_mt,$(RELEASEDIR.tbb.libia))
endif
releasetbb.LIBS_A := $(if $(OS_is_win),$(TBBDIR.libia)/tbb12$(dtbb).$(a) $(TBBDIR.libia)/tbbmalloc$(dtbb).$(a))
releasetbb.LIBS_Y := $(TBBDIR.soia)/$(plib)tbb$(if $(OS_is_win),12$(dtbb),).$(y) $(TBBDIR.soia)/$(plib)tbbmalloc$(dtbb).$(y)                                                           \
                     $(if $(OS_is_lnx), $(if $(wildcard $(TBBDIR.soia)/libtbbmalloc.so.2),$(wildcard $(TBBDIR.soia)/libtbbmalloc.so.2))\
                                                            $(if $(wildcard $(TBBDIR.soia)/libtbbmalloc.so.12),$(wildcard $(TBBDIR.soia)/libtbbmalloc.so.12))\
                                                            $(if $(wildcard $(TBBDIR.soia)/libtbb.so.2),$(wildcard $(TBBDIR.soia)/libtbb.so.2))\
                                                            $(if $(wildcard $(TBBDIR.soia)/libtbb.so.12),$(wildcard $(TBBDIR.soia)/libtbb.so.12))) \
                     $(if $(OS_is_mac),$(if $(wildcard $(TBBDIR.soia)/libtbb.12.dylib),$(wildcard $(TBBDIR.soia)/libtbb.12.dylib))\
                                       $(if $(wildcard $(TBBDIR.soia)/libtbbmalloc.2.dylib),$(wildcard $(TBBDIR.soia)/libtbbmalloc.2.dylib)))


#============================= Math backend folders =====================================

ifeq ($(BACKEND_CONFIG), ref)
    ifeq ($(RNG_BACKEND), openrng)
        RNG_OPENRNG := yes
    endif
    ifndef RNG_BACKEND
        RNG_BACKEND := ref
    endif
    $(if $(filter $(RNG_BACKEND),ref openrng),,$(error unknown rng backend $(RNG_BACKEND)))
endif

ifeq ($(BACKEND_CONFIG), mkl)
    ifndef RNG_BACKEND
        RNG_BACKEND := mkl
    endif
    $(if $(filter $(RNG_BACKEND),mkl),,$(error mkl backend does not support the rng backend $(RNG_BACKEND)))
endif

include dev/make/deps.$(BACKEND_CONFIG).mk

#=============================== VTune SDK folders ======================================

ifeq ($(REQPROFILE), yes)
    -DPROFILER := -DONEDAL_KERNEL_PROFILER
    VTUNESDK.include := $(VTUNE_PROFILER_DIR)/sdk/include
    VTUNESDK.libia := $(if $(OS_is_lnx), $(VTUNE_PROFILER_DIR)/sdk/lib64,)
    VTUNESDK.LIBS_A := $(if $(OS_is_lnx), $(VTUNESDK.libia)/libittnotify.a,)
endif

#===============================================================================
# Release library names
#===============================================================================
include makefile.ver

dep_thr := $(if $(MSVC_RT_is_release),tbb12.lib tbbmalloc.lib msvcrt.lib msvcprt.lib /nodefaultlib:libucrt.lib ucrt.lib, tbb12_debug.lib tbbmalloc_debug.lib msvcrtd.lib msvcprtd.lib /nodefaultlib:libucrtd.lib ucrtd.lib)
dep_seq := $(if $(MSVC_RT_is_release),msvcrt.lib msvcprt.lib, msvcrtd.lib msvcprtd.lib)

y_full_name_postfix := $(if $(OS_is_win),,$(if $(OS_is_mac),.$(MAJORBINARY).$(MINORBINARY).$(y),.$(y).$(MAJORBINARY).$(MINORBINARY)))
y_major_name_postfix := $(if $(OS_is_win),,$(if $(OS_is_mac),.$(MAJORBINARY).$(y),.$(y).$(MAJORBINARY)))

core_a       := $(plib)onedal_core$d.$a
core_y       := $(plib)onedal_core$d$(if $(OS_is_win),.$(MAJORBINARY),).$y
oneapi_a     := $(plib)onedal$d.$a
oneapi_y     := $(plib)onedal$d$(if $(OS_is_win),.$(MAJORBINARY),).$y
oneapi_a.dpc := $(plib)onedal_dpc$d.$a
oneapi_y.dpc := $(plib)onedal_dpc$d$(if $(OS_is_win),.$(MAJORBINARY),).$y
parameters_a     := $(plib)onedal_parameters$d.$a
parameters_y     := $(plib)onedal_parameters$d$(if $(OS_is_win),.$(MAJORBINARY),).$y
parameters_a.dpc := $(plib)onedal_parameters_dpc$d.$a
parameters_y.dpc := $(plib)onedal_parameters_dpc$d$(if $(OS_is_win),.$(MAJORBINARY),).$y

thr_tbb_a := $(plib)onedal_thread$d.$a
thr_tbb_y := $(plib)onedal_thread$d$(if $(OS_is_win),.$(MAJORBINARY),).$y

release.LIBS_A := $(core_a) \
                  $(if $(OS_is_win),$(foreach ilib,$(core_a),$(ilib:%.lib=%_dll.lib)),) \
                  $(if $(DAALTHRS),$(foreach i,$(DAALTHRS),$(thr_$(i)_a)),)
release.LIBS_Y := $(core_y) \
                  $(if $(DAALTHRS),$(foreach i,$(DAALTHRS),$(thr_$(i)_y)),)

release.ONEAPI.LIBS_A := $(oneapi_a) \
                         $(if $(OS_is_win),$(foreach ilib,$(oneapi_a),$(ilib:%.lib=%_dll.lib)),)
release.ONEAPI.LIBS_Y := $(oneapi_y)

release.ONEAPI.LIBS_A.dpc := $(oneapi_a.dpc) \
                             $(if $(OS_is_win),$(foreach ilib,$(oneapi_a.dpc),$(ilib:%.lib=%_dll.lib)),)
release.ONEAPI.LIBS_Y.dpc := $(oneapi_y.dpc)

release.PARAMETERS.LIBS_A := $(parameters_a) \
                         $(if $(OS_is_win),$(foreach ilib,$(parameters_a),$(ilib:%.lib=%_dll.lib)),)
release.PARAMETERS.LIBS_Y := $(parameters_y)

release.PARAMETERS.LIBS_A.dpc := $(parameters_a.dpc) \
                             $(if $(OS_is_win),$(foreach ilib,$(parameters_a.dpc),$(ilib:%.lib=%_dll.lib)),)
release.PARAMETERS.LIBS_Y.dpc := $(parameters_y.dpc)


$(eval $(call set_daal_rt_deps))

daaldep.rt.thr  := $(daaldep.$(PLAT).rt.thr)
daaldep.rt.seq  := $(daaldep.$(PLAT).rt.seq)
daaldep.rt.dpc  := $(daaldep.$(PLAT).rt.dpc)

# List oneAPI header files to populate release/include.
release.ONEAPI.HEADERS.exclude := ! -path "*/backend/*" ! -path "*.impl.*" ! -path "*_test.*" ! -path "*/test/*"
release.ONEAPI.HEADERS := $(shell find $(CPPDIR) -type f -name "*.hpp" $(release.ONEAPI.HEADERS.exclude))
release.ONEAPI.HEADERS.OSSPEC := $(foreach fn,$(release.ONEAPI.HEADERS),$(if $(filter %$(_OS),$(basename $(fn))),$(fn)))
release.ONEAPI.HEADERS.COMMON := $(foreach fn,$(release.ONEAPI.HEADERS),$(if $(filter $(addprefix %,$(OSList)),$(basename $(fn))),,$(fn)))
release.ONEAPI.HEADERS.COMMON := $(filter-out $(subst _$(_OS),,$(release.ONEAPI.HEADERS.OSSPEC)),$(release.ONEAPI.HEADERS.COMMON))

# List header files to populate release/include.
release.HEADERS := $(shell find $(CPPDIR.daal)/include -type f -name "*.h")
release.HEADERS.OSSPEC := $(foreach fn,$(release.HEADERS),$(if $(filter %$(_OS),$(basename $(fn))),$(fn)))
release.HEADERS.COMMON := $(foreach fn,$(release.HEADERS),$(if $(filter $(addprefix %,$(OSList)),$(basename $(fn))),,$(fn)))
release.HEADERS.COMMON := $(filter-out $(subst _$(_OS),,$(release.HEADERS.OSSPEC)),$(release.HEADERS.COMMON))

# List examples files to populate release/examples.
expat = %.cpp %.h %.hpp %.txt %.csv %.cmake
expat += $(if $(OS_is_win),%.bat,%_$(_OS).lst %_$(_OS).sh)
release.EXAMPLES.CMAKE := $(filter $(expat),$(shell find examples/cmake -type f))
release.EXAMPLES.CPP   := $(filter $(expat),$(shell find examples/daal/cpp  -type f))
release.EXAMPLES.DATA  := $(filter $(expat),$(shell find examples/daal/data -type f))
release.ONEAPI.EXAMPLES.CPP  := $(filter $(expat),$(shell find examples/oneapi/cpp -type f))
release.ONEAPI.EXAMPLES.DPC  := $(filter $(expat),$(shell find examples/oneapi/dpc -type f))
release.ONEAPI.EXAMPLES.DATA := $(filter $(expat),$(shell find examples/oneapi/data -type f))

# List env files to populate release.
release.ENV = deploy/local/vars_$(_OS).$(scr)

# List modulefiles to populate release.
release.MODULEFILES = deploy/local/dal

# List config files to populate release.
release.CONF = deploy/local/config.txt

# List samples files to populate release/examples.
SAMPLES.srcdir:= $(DIR)/samples
spat = %.scala %.cpp %.h %.hpp %.txt %.csv %.html %.png %.parquet %.blob %.cmake
spat += $(if $(OS_is_win),%.bat,%_$(_OS).lst %makefile_$(_OS) %.sh)
release.SAMPLES.CMAKE := $(filter $(spat),$(shell find $(SAMPLES.srcdir)/cmake -type f))
release.SAMPLES.CPP  := $(if $(wildcard $(SAMPLES.srcdir)/daal/cpp/*),                                                   \
                          $(if $(OS_is_mac),                                                                             \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/daal/cpp -not -wholename '*mpi*' -type f))   \
                          ,                                                                                              \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/daal/cpp -type f))                           \
                          )                                                                                              \
                        )
release.SAMPLES.ONEDAL.DPC  := $(if $(wildcard $(SAMPLES.srcdir)/oneapi/dpc/*),                                          \
                          $(if $(OS_is_mac),                                                                             \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/oneapi/dpc -not -wholename '*mpi*' -type f)) \
                          ,                                                                                              \
                            $(filter $(spat),$(shell find $(SAMPLES.srcdir)/oneapi/dpc -type f))                         \
                          )                                                                                              \
                        )

# List doc files to populate release/documentation.
DOC.srcdir:= $(DIR)/../documentation
release.DOC := $(shell if [ -d $(DOC.srcdir) ]; then find $(DOC.srcdir) -not -wholename '*.svn*' -type f ;fi)
release.DOC.COMMON := $(foreach fn,$(release.DOC),$(if $(filter $(addprefix %,$(OSList)),$(basename $(fn))),,$(fn)))
release.DOC.OSSPEC := $(foreach fn,$(release.DOC),$(if $(filter %$(_OS),$(basename $(fn))),$(fn)))

#===============================================================================
# Core part
#===============================================================================
include makefile.lst

THR.srcdir       := $(CPPDIR.daal)/src/threading
CORE.srcdir      := $(CPPDIR.daal)/src/algorithms
EXTERNALS.srcdir := $(CPPDIR.daal)/src/externals

CORE.SERV.srcdir := $(subst \,/,$(CPPDIR.daal)/src/services)
CORE.SERV.COMPILER.srcdir := $(CPPDIR.daal)/src/services/compiler/$(CORE.SERV.COMPILER.$(COMPILER))

CORE.srcdirs  := $(CORE.SERV.srcdir) $(CORE.srcdir)                  \
                 $(if $(DAALTHRS),,$(THR.srcdir))                    \
                 $(addprefix $(CORE.SERV.srcdir)/, $(CORE.SERVICES)) \
                 $(addprefix $(CORE.srcdir)/, $(CORE.ALGORITHMS))    \
                 $(CORE.SERV.COMPILER.srcdir) $(EXTERNALS.srcdir)    \
                 $(CPPDIR.daal)/src/sycl \
                 $(CPPDIR.daal)/src/data_management

CORE.incdirs.common := $(RELEASEDIR.include) $(CPPDIR.daal) $(WORKDIR)
CORE.incdirs.thirdp := $(daaldep.math_backend.incdir) $(VTUNESDK.include) $(TBBDIR.include)
CORE.incdirs := $(CORE.incdirs.common) $(CORE.incdirs.thirdp)

$(info CORE.incdirs: $(CORE.incdirs))

containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
notcontaining = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
cpy = cp -fp "$<" "$@"

CORE.tmpdir_a := $(WORKDIR)/core_static
CORE.tmpdir_y := $(WORKDIR)/core_dynamic
CORE.srcs     := $(notdir $(wildcard $(CORE.srcdirs:%=%/*.cpp)))
CORE.srcs     := $(if $(OS_is_mac),$(CORE.srcs),$(call notcontaining,_mac,$(CORE.srcs)))
CORE.objs_a   := $(CORE.srcs:%.cpp=$(CORE.tmpdir_a)/%.$o)
CORE.objs_a   := $(filter-out %core_threading_win_dll.$o,$(CORE.objs_a))
CORE.objs_y   := $(CORE.srcs:%.cpp=$(CORE.tmpdir_y)/%.$o)
CORE.objs_y   := $(if $(OS_is_win),$(CORE.objs_y),$(filter-out %core_threading_win_dll.$o,$(CORE.objs_y)))

CORE.objs_a_tmp := $(call containing,_fpt,$(CORE.objs_a))
CORE.objs_a     := $(call notcontaining,_fpt,$(CORE.objs_a))
CORE.objs_a_tpl := $(subst _fpt,_fpt_flt,$(CORE.objs_a_tmp)) $(subst _fpt,_fpt_dbl,$(CORE.objs_a_tmp))
CORE.objs_a     := $(CORE.objs_a) $(CORE.objs_a_tpl)

CORE.objs_a_tmp := $(call containing,_cpu,$(CORE.objs_a))
CORE.objs_a     := $(call notcontaining,_cpu,$(CORE.objs_a))
CORE.objs_a_tpl := $(foreach ccc,$(USECPUS.files),$(subst _cpu,_cpu_$(ccc),$(CORE.objs_a_tmp)))
CORE.objs_a     := $(CORE.objs_a) $(CORE.objs_a_tpl)

CORE.objs_y_tmp := $(call containing,_fpt,$(CORE.objs_y))
CORE.objs_y     := $(call notcontaining,_fpt,$(CORE.objs_y))
CORE.objs_y_tpl := $(subst _fpt,_fpt_flt,$(CORE.objs_y_tmp)) $(subst _fpt,_fpt_dbl,$(CORE.objs_y_tmp))
CORE.objs_y     := $(CORE.objs_y) $(CORE.objs_y_tpl)

CORE.objs_y_tmp := $(call containing,_cpu,$(CORE.objs_y))
CORE.objs_y     := $(call notcontaining,_cpu,$(CORE.objs_y))
CORE.objs_y_tpl := $(foreach ccc,$(USECPUS.files),$(subst _cpu,_cpu_$(ccc),$(CORE.objs_y_tmp)))
CORE.objs_y     := $(CORE.objs_y) $(CORE.objs_y_tpl)

-include $(CORE.tmpdir_a)/*.d
-include $(CORE.tmpdir_y)/*.d

$(CORE.tmpdir_a)/$(core_a:%.$a=%_link.txt): $(CORE.objs_a) | $(CORE.tmpdir_a)/. ; $(WRITE.PREREQS)
$(CORE.tmpdir_a)/$(core_a:%.$a=%_link.$a):  LOPT:=
$(CORE.tmpdir_a)/$(core_a:%.$a=%_link.$a):  $(CORE.tmpdir_a)/$(core_a:%.$a=%_link.txt) | $(CORE.tmpdir_a)/. ; $(LINK.STATIC)
$(WORKDIR.lib)/$(core_a):                   LOPT:=
$(WORKDIR.lib)/$(core_a):                   $(daaldep.math_backend.ext) $(VTUNESDK.LIBS_A) $(CORE.tmpdir_a)/$(core_a:%.$a=%_link.$a) ; $(LINK.STATIC)

$(WORKDIR.lib)/$(core_y): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(core_y): LOPT += $(daaldep.rt.seq)
$(WORKDIR.lib)/$(core_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
ifdef OS_is_win
$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(core_y)
endif
$(CORE.tmpdir_y)/$(core_y:%.$y=%_link.txt): $(CORE.objs_y) $(if $(OS_is_win),$(CORE.tmpdir_y)/dll.res,) | $(CORE.tmpdir_y)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(core_y):                   $(daaldep.math_backend.ext) $(VTUNESDK.LIBS_A) \
                                            $(CORE.tmpdir_y)/$(core_y:%.$y=%_link.txt) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

$(CORE.objs_a): $(CORE.tmpdir_a)/inc_a_folders.txt
$(CORE.objs_a): COPT += $(-fPIC) $(-cxx17) $(-Zl) $(-DEBC) $(-DMKL_ILP64) $(-DPROFILER)
$(CORE.objs_a): COPT += -D__TBB_NO_IMPLICIT_LINKAGE -DDAAL_NOTHROW_EXCEPTIONS \
                        -DDAAL_HIDE_DEPRECATED -DTBB_USE_ASSERT=0 -D_ENABLE_ATOMIC_ALIGNMENT_FIX \
                        $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG)
$(CORE.objs_a): COPT += @$(CORE.tmpdir_a)/inc_a_folders.txt

$(eval $(call append_uarch_copt,$(CORE.objs_a)))

$(CORE.objs_y): $(CORE.tmpdir_y)/inc_y_folders.txt
$(CORE.objs_y): COPT += $(-fPIC) $(-cxx17) $(-Zl) $(-DEBC) $(-DMKL_ILP64) $(-DPROFILER)
$(CORE.objs_y): COPT += -D__DAAL_IMPLEMENTATION \
                        -D__TBB_NO_IMPLICIT_LINKAGE -DDAAL_NOTHROW_EXCEPTIONS \
                        -DDAAL_HIDE_DEPRECATED -DTBB_USE_ASSERT=0 -D_ENABLE_ATOMIC_ALIGNMENT_FIX \
                        $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG)
$(CORE.objs_y): COPT += @$(CORE.tmpdir_y)/inc_y_folders.txt

$(eval $(call append_uarch_copt,$(CORE.objs_y)))

vpath
vpath %.cpp $(CORE.srcdirs)
vpath %.rc $(CORE.srcdirs)

$(CORE.tmpdir_a)/inc_a_folders.txt: makefile.lst | $(CORE.tmpdir_a)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))
$(CORE.tmpdir_y)/inc_y_folders.txt: makefile.lst | $(CORE.tmpdir_y)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))

$(CORE.tmpdir_a)/library_version_info.$(o): $(VERSION_DATA_FILE)
$(CORE.tmpdir_y)/library_version_info.$(o): $(VERSION_DATA_FILE)

# Used as $(eval $(call .compile.template.ay,obj_file))
define .compile.template.ay
$(eval template_source_cpp := $(subst .$o,.cpp,$(notdir $1)))
$(eval template_source_cpp := $(subst _fpt_flt,_fpt,$(template_source_cpp)))
$(eval template_source_cpp := $(subst _fpt_dbl,_fpt,$(template_source_cpp)))

$(eval $(call subst_arch_cpu_in_var,template_source_cpp))

$1: $(template_source_cpp) ; $(value C.COMPILE)
endef

$(foreach a,$(CORE.objs_a),$(eval $(call .compile.template.ay,$a)))
$(foreach a,$(CORE.objs_y),$(eval $(call .compile.template.ay,$a)))

$(CORE.tmpdir_y)/dll.res: $(VERSION_DATA_FILE)
$(CORE.tmpdir_y)/dll.res: RCOPT += $(addprefix -I, $(CORE.incdirs.common))
$(CORE.tmpdir_y)/%.res: %.rc | $(CORE.tmpdir_y)/. ; $(RC.COMPILE)


#===============================================================================
# oneAPI part
#===============================================================================
ONEAPI.tmpdir_a := $(WORKDIR)/oneapi_static
ONEAPI.tmpdir_y := $(WORKDIR)/oneapi_dynamic
PARAMETERS.tmpdir_a := $(WORKDIR)/parameters_static
PARAMETERS.tmpdir_y := $(WORKDIR)/parameters_dynamic
ONEAPI.tmpdir_a.dpc := $(WORKDIR)/oneapi_dpc_static
ONEAPI.tmpdir_y.dpc := $(WORKDIR)/oneapi_dpc_dynamic
PARAMETERS.tmpdir_a.dpc := $(WORKDIR)/parameters_dpc_static
PARAMETERS.tmpdir_y.dpc := $(WORKDIR)/parameters_dpc_dynamic

ONEAPI.incdirs.common := $(CPPDIR)
ONEAPI.incdirs.thirdp := $(CORE.incdirs.common) $(daaldep.math_backend_oneapi.incdir) $(VTUNESDK.include) $(TBBDIR.include)
ONEAPI.incdirs := $(ONEAPI.incdirs.common) $(CORE.incdirs.thirdp) $(ONEAPI.incdirs.thirdp)

ONEAPI.dispatcher_cpu = $(WORKDIR)/oneapi/dal/_dal_cpu_dispatcher_gen.hpp

ONEAPI.srcdir := $(CPPDIR.onedal)
ONEAPI.srcdirs.base := $(ONEAPI.srcdir) \
                       $(ONEAPI.srcdir)/algo \
                       $(ONEAPI.srcdir)/table \
                       $(ONEAPI.srcdir)/spmd \
                       $(ONEAPI.srcdir)/graph \
                       $(ONEAPI.srcdir)/util \
                       $(ONEAPI.srcdir)/io \
                       $(addprefix $(ONEAPI.srcdir)/algo/, $(ONEAPI.ALGOS)) \
                       $(addprefix $(ONEAPI.srcdir)/io/, $(ONEAPI.IO))
ONEAPI.srcdirs.detail := $(foreach x,$(ONEAPI.srcdirs.base),$(shell find $x -maxdepth 1 -type d -name detail))
ONEAPI.srcdirs.backend := $(foreach x,$(ONEAPI.srcdirs.base),$(shell find $x -maxdepth 1 -type d -name backend))
ONEAPI.srcdirs.parameters := $(ONEAPI.srcdir)/detail/parameters \
                             $(foreach x,$(ONEAPI.srcdirs.base),$(shell find $x -maxdepth 1 -type d -name parameters))
ONEAPI.srcdirs := $(ONEAPI.srcdirs.base) $(ONEAPI.srcdirs.detail) $(ONEAPI.srcdirs.backend) $(ONEAPI.srcdirs.parameters)

ONEAPI.srcs.all.exclude := ! -path "*_test.*" ! -path "*/test/*" ! -path "*/detail/parameters/*"
ONEAPI.srcs.parameters.exclude := ! -path "*_test.*" ! -path "*/test/*"
ONEAPI.srcs.all := $(foreach x,$(ONEAPI.srcdirs.base),$(shell find $x -maxdepth 1 -type f -name "*.cpp" $(ONEAPI.srcs.all.exclude))) \
                   $(foreach x,$(ONEAPI.srcdirs.detail),$(shell find $x -type f -name "*.cpp" $(ONEAPI.srcs.all.exclude))) \
                   $(foreach x,$(ONEAPI.srcdirs.backend),$(shell find $x -type f -name "*.cpp" $(ONEAPI.srcs.all.exclude))) \
                   $(foreach x,$(ONEAPI.srcdirs.parameters),$(shell find $x -type f -name "*.cpp" $(ONEAPI.srcs.parameters.exclude)))
ONEAPI.srcs.all	:= $(ONEAPI.srcs.all:./%=%)
ONEAPI.srcs.dpc := $(filter %_dpc.cpp,$(ONEAPI.srcs.all))
ONEAPI.srcs     := $(filter-out %_dpc.cpp,$(ONEAPI.srcs.all))
ONEAPI.srcs.dpc := $(ONEAPI.srcs) $(ONEAPI.srcs.dpc)

ONEAPI.srcs.mangled     := $(subst /,-,$(ONEAPI.srcs))
ONEAPI.srcs.mangled.dpc := $(subst /,-,$(ONEAPI.srcs.dpc))

ONEAPI.objs_a     := $(ONEAPI.srcs.mangled:%.cpp=$(ONEAPI.tmpdir_a)/%.$o)
ONEAPI.objs_y     := $(ONEAPI.srcs.mangled:%.cpp=$(ONEAPI.tmpdir_y)/%.$o)
ONEAPI.objs_a.dpc := $(ONEAPI.srcs.mangled.dpc:%.cpp=$(ONEAPI.tmpdir_a.dpc)/%.$o)
ONEAPI.objs_y.dpc := $(ONEAPI.srcs.mangled.dpc:%.cpp=$(ONEAPI.tmpdir_y.dpc)/%.$o)
ONEAPI.objs_a.all := $(ONEAPI.objs_a) $(ONEAPI.objs_a.dpc)
ONEAPI.objs_y.all := $(ONEAPI.objs_y) $(ONEAPI.objs_y.dpc)

# Populate _cpu files -> _cpu_%cpu_name%, where %cpu_name% is $(USECPUS.files)
# $1 Output variable name
# $2 List of object files
define .populate_cpus
$(eval non_cpu_files := $(call notcontaining,_cpu,$2))
$(eval cpu_files := $(call containing,_cpu,$2))

$(eval $(call add_cpu_to_uarch_in_files,non_cpu_files))

$(eval populated_cpu_files := $(foreach ccc,$(USECPUS.files),$(subst _cpu,_cpu_$(ccc),$(cpu_files))))
$(eval populated_cpu_files := $(filter-out $(user_cpu_files),$(populated_cpu_files)))
$(eval $1 := $(non_cpu_files) $(populated_cpu_files))
endef

$(eval $(call .populate_cpus,ONEAPI.objs_a,$(ONEAPI.objs_a)))
$(eval $(call .populate_cpus,ONEAPI.objs_y,$(ONEAPI.objs_y)))
$(eval $(call .populate_cpus,ONEAPI.objs_a.dpc,$(ONEAPI.objs_a.dpc)))
$(eval $(call .populate_cpus,ONEAPI.objs_y.dpc,$(ONEAPI.objs_y.dpc)))

-include $(ONEAPI.tmpdir_a)/*.d
-include $(ONEAPI.tmpdir_y)/*.d
-include $(ONEAPI.tmpdir_a.dpc)/*.d
-include $(ONEAPI.tmpdir_y.dpc)/*.d

# Declares target for object file compilation
# $1: Object file
# $2: Temporary directory where object file is stored
# $3: Compiler id (C or DPC)
define .ONEAPI.compile
$(eval template_source_cpp := $(1:$2/%.$o=%.cpp))
$(eval template_source_cpp := $(subst -,/,$(template_source_cpp)))

$(eval $(call subst_arch_cpu_in_var,template_source_cpp))

$1: $(template_source_cpp) | $(dir $1)/. ; $(value $3.COMPILE)
endef

# Declares target to compile static library
# $1: Path to the static library to be produced
# $2: List of dependencies
define .ONEAPI.declare_static_lib
$(1:%.$a=%_link.txt): $2 | $(dir $1)/. ; $(value WRITE.PREREQS)
$1: LOPT:=
$1: $(1:%.$a=%_link.txt) | $(dir $1)/. ; $(value LINK.STATIC)
endef

$(eval $(call dispatcher_cpu_rule,$(ONEAPI.dispatcher_cpu),$(USECPUS)))

# Create file with include paths
ONEAPI.include_options := $(addprefix -I, $(ONEAPI.incdirs.common)) \
                          $(addprefix $(-isystem), $(ONEAPI.incdirs.thirdp))

$(ONEAPI.tmpdir_a)/inc_a_folders.txt: | $(ONEAPI.tmpdir_a)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

$(ONEAPI.tmpdir_y)/inc_y_folders.txt: | $(ONEAPI.tmpdir_y)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

$(ONEAPI.tmpdir_a.dpc)/inc_a_folders.txt: | $(ONEAPI.tmpdir_a.dpc)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

$(ONEAPI.tmpdir_y.dpc)/inc_y_folders.txt: | $(ONEAPI.tmpdir_y.dpc)/.
	$(call WRITE.PREREQS,$(ONEAPI.include_options),$(space))

# Set compilation options to the object files which are part of STATIC lib
$(ONEAPI.objs_a): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_a)/inc_a_folders.txt
$(ONEAPI.objs_a): COPT += $(-fPIC) $(-cxx17) $(-Zl) $(-DMKL_ILP64) $(-DEBC) $(-EHsc) $(pedantic.opts) \
                          -DDAAL_NOTHROW_EXCEPTIONS \
                          -DDAAL_HIDE_DEPRECATED \
                          -D_ENABLE_ATOMIC_ALIGNMENT_FIX \
                          -D__TBB_NO_IMPLICIT_LINKAGE \
                          -DTBB_USE_ASSERT=0 \
                           @$(ONEAPI.tmpdir_a)/inc_a_folders.txt

$(eval $(call update_copt_from_dispatcher_tag,$(ONEAPI.objs_a)))

$(ONEAPI.objs_a.dpc): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_a.dpc)/inc_a_folders.txt
$(ONEAPI.objs_a.dpc): COPT += $(-fPIC) $(-cxx17) $(-Zl_DPCPP) $(-DMKL_ILP64) $(-DEBC_DPCPP) $(-EHsc) $(pedantic.opts.dpcpp) \
                              -DDAAL_NOTHROW_EXCEPTIONS \
                              -DDAAL_HIDE_DEPRECATED \
                              -DONEDAL_DATA_PARALLEL \
                              -D__TBB_NO_IMPLICIT_LINKAGE \
                              -D_ENABLE_ATOMIC_ALIGNMENT_FIX \
                              -DTBB_USE_ASSERT=0 \
                               @$(ONEAPI.tmpdir_a.dpc)/inc_a_folders.txt

$(eval $(call update_copt_from_dispatcher_tag,$(ONEAPI.objs_a.dpc),.dpcpp))

# Set compilation options to the object files which are part of DYNAMIC lib
$(ONEAPI.objs_y): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_y)/inc_y_folders.txt
$(ONEAPI.objs_y): COPT += $(-fPIC) $(-cxx17) $(-Zl) $(-DMKL_ILP64) $(-DEBC) $(-EHsc) $(pedantic.opts) \
                          -DDAAL_NOTHROW_EXCEPTIONS \
                          -DDAAL_HIDE_DEPRECATED \
                          -D_ENABLE_ATOMIC_ALIGNMENT_FIX \
                          $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG) \
                          -D__ONEDAL_ENABLE_DLL_EXPORT__ \
                          -D__TBB_NO_IMPLICIT_LINKAGE \
                          -DTBB_USE_ASSERT=0 \
                          @$(ONEAPI.tmpdir_y)/inc_y_folders.txt

$(eval $(call update_copt_from_dispatcher_tag,$(ONEAPI.objs_y)))

# Note: The libonedal_dpc.so library does not support debug mode.
# When compiling with the debug flag $(-DEBC_DPCPP), linking with libonedal_dpc.so may cause indefinite linking times
# due to the extensive processing of debug information. For debugging, please use the static library version (libonedal_dpc.a).
$(ONEAPI.objs_y.dpc): $(ONEAPI.dispatcher_cpu) $(ONEAPI.tmpdir_y.dpc)/inc_y_folders.txt
$(ONEAPI.objs_y.dpc): COPT += $(-fPIC) $(-cxx17) $(-Zl_DPCPP) $(-DMKL_ILP64) $(-EHsc) $(pedantic.opts.dpcpp) \
                              -DDAAL_NOTHROW_EXCEPTIONS \
                              -DDAAL_HIDE_DEPRECATED \
                              -DONEDAL_DATA_PARALLEL \
                              -D_ENABLE_ATOMIC_ALIGNMENT_FIX \
                              $(if $(CHECK_DLL_SIG),-DDAAL_CHECK_DLL_SIG) \
                              -D__ONEDAL_ENABLE_DLL_EXPORT__ \
                              -D__TBB_NO_IMPLICIT_LINKAGE \
                              -DTBB_USE_ASSERT=0 \
                              @$(ONEAPI.tmpdir_y.dpc)/inc_y_folders.txt

$(eval $(call update_copt_from_dispatcher_tag,$(ONEAPI.objs_y.dpc),.dpcpp))

# Filtering parameter files
PARAMETERS.objs_a.filtered := $(filter %parameters.$(o) %parameters_impl.$(o),$(ONEAPI.objs_a))
ONEAPI.objs_a.filtered := $(filter-out %parameters.$(o) %parameters_impl.$(o),$(ONEAPI.objs_a))
PARAMETERS.objs_y.filtered := $(filter %parameters.$(o) %parameters_impl.$(o),$(ONEAPI.objs_y))
ONEAPI.objs_y.filtered := $(filter-out %parameters.$(o) %parameters_impl.$(o),$(ONEAPI.objs_y))
PARAMETERS.objs_a.dpc.filtered := $(filter %parameters.$(o) %parameters_impl.$(o) %parameters_dpc.$(o),$(ONEAPI.objs_a.dpc))
ONEAPI.objs_a.dpc.filtered := $(filter-out %parameters.$(o) %parameters_impl.$(o) %parameters_dpc.$(o),$(ONEAPI.objs_a.dpc))
PARAMETERS.objs_y.dpc.filtered := $(filter %parameters.$(o) %parameters_impl.$(o) %parameters_dpc.$(o),$(ONEAPI.objs_y.dpc))
ONEAPI.objs_y.dpc.filtered := $(filter-out %parameters.$(o) %parameters_impl.$(o) %parameters_dpc.$(o),$(ONEAPI.objs_y.dpc))

# Actual compilation
$(foreach x,$(ONEAPI.objs_a.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_a),C)))
$(foreach x,$(ONEAPI.objs_y.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_y),C)))
$(foreach x,$(PARAMETERS.objs_a.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_a),C)))
$(foreach x,$(PARAMETERS.objs_y.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_y),C)))
$(foreach x,$(ONEAPI.objs_a.dpc.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_a.dpc),DPC)))
$(foreach x,$(ONEAPI.objs_y.dpc.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_y.dpc),DPC)))
$(foreach x,$(PARAMETERS.objs_a.dpc.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_a.dpc),DPC)))
$(foreach x,$(PARAMETERS.objs_y.dpc.filtered),$(eval $(call .ONEAPI.compile,$x,$(ONEAPI.tmpdir_y.dpc),DPC)))

# Create Host and DPC++ oneapi libraries
ifeq ($(BUILD_PARAMETERS_LIB),yes)
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(oneapi_a),$(ONEAPI.objs_a.filtered)))
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(oneapi_a.dpc),$(ONEAPI.objs_a.dpc.filtered)))
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(parameters_a),$(PARAMETERS.objs_a.filtered)))
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(parameters_a.dpc),$(PARAMETERS.objs_a.dpc.filtered)))
else
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(oneapi_a),$(ONEAPI.objs_a)))
$(eval $(call .ONEAPI.declare_static_lib,$(WORKDIR.lib)/$(oneapi_a.dpc),$(ONEAPI.objs_a.dpc)))
endif

ONEAPI.objs_y.lib := $(ONEAPI.objs_y.filtered)
ifeq ($(BUILD_PARAMETERS_LIB),no)
  ONEAPI.objs_y.lib += $(PARAMETERS.objs_y.filtered)
endif

$(ONEAPI.tmpdir_y)/$(oneapi_y:%.$y=%_link.txt): \
    $(ONEAPI.objs_y.lib) $(if $(OS_is_win),$(ONEAPI.tmpdir_y)/dll.res,) | $(ONEAPI.tmpdir_y)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(oneapi_y): \
    $(daaldep.math_backend.ext) \
    $(ONEAPI.tmpdir_y)/$(oneapi_y:%.$y=%_link.txt) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(daaldep.rt.seq)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
$(WORKDIR.lib)/$(oneapi_y): LOPT += $(if $(OS_is_win),$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib))
ifdef OS_is_win
$(WORKDIR.lib)/$(oneapi_y:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(oneapi_y)
endif

ifeq ($(BUILD_PARAMETERS_LIB),yes)
$(ONEAPI.tmpdir_y)/$(parameters_y:%.$y=%_link.txt): \
    $(PARAMETERS.objs_y.filtered) $(if $(OS_is_win),$(ONEAPI.tmpdir_y)/dll.res,) | $(ONEAPI.tmpdir_y)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(parameters_y): \
    $(WORKDIR.lib)/$(oneapi_y) $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) \
    $(ONEAPI.tmpdir_y)/$(parameters_y:%.$y=%_link.txt) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)
$(WORKDIR.lib)/$(parameters_y): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(parameters_y): LOPT += $(daaldep.rt.seq)
$(WORKDIR.lib)/$(parameters_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
$(WORKDIR.lib)/$(parameters_y): LOPT += $(if $(OS_is_win),$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib))
ifdef OS_is_win
$(WORKDIR.lib)/$(parameters_y:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(parameters_y)
endif
endif

ONEAPI.objs_y.dpc.lib := $(ONEAPI.objs_y.dpc.filtered)
ifeq ($(BUILD_PARAMETERS_LIB),no)
  ONEAPI.objs_y.dpc.lib += $(PARAMETERS.objs_y.dpc.filtered)
endif

$(ONEAPI.tmpdir_y.dpc)/$(oneapi_y.dpc:%.$y=%_link.txt): \
    $(ONEAPI.objs_y.dpc.lib) $(if $(OS_is_win),$(ONEAPI.tmpdir_y.dpc)/dll.res,) | $(ONEAPI.tmpdir_y.dpc)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(oneapi_y.dpc): \
    $(daaldep.math_backend.ext) \
    $(ONEAPI.tmpdir_y.dpc)/$(oneapi_y.dpc:%.$y=%_link.txt) ; $(DPC.LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(daaldep.rt.dpc)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(if $(REQDBG),-flink-huge-device-code,)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(if $(OS_is_win),$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib))
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(if $(OS_is_win),sycl$d.lib OpenCL.lib)
$(WORKDIR.lib)/$(oneapi_y.dpc): LOPT += $(daaldep.math_backend.sycl)

ifdef OS_is_win
$(WORKDIR.lib)/$(oneapi_y.dpc:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(oneapi_y.dpc)
endif

ifeq ($(BUILD_PARAMETERS_LIB),yes)
$(ONEAPI.tmpdir_y.dpc)/$(parameters_y.dpc:%.$y=%_link.txt): \
    $(PARAMETERS.objs_y.dpc.filtered) $(if $(OS_is_win),$(ONEAPI.tmpdir_y.dpc)/dll.res,) | $(ONEAPI.tmpdir_y.dpc)/. ; $(WRITE.PREREQS)
$(WORKDIR.lib)/$(parameters_y.dpc): \
    $(WORKDIR.lib)/$(oneapi_y.dpc) $(daaldep.ipp) $(daaldep.vml) $(daaldep.mkl) \
    $(ONEAPI.tmpdir_y.dpc)/$(parameters_y.dpc:%.$y=%_link.txt) ; $(DPC.LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)
$(WORKDIR.lib)/$(parameters_y.dpc): LOPT += $(-fPIC)
$(WORKDIR.lib)/$(parameters_y.dpc): LOPT += $(daaldep.rt.dpc)
$(WORKDIR.lib)/$(parameters_y.dpc): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.$(MAJORBINARY).dll=%_dll.lib),)
$(WORKDIR.lib)/$(parameters_y.dpc): LOPT += $(if $(OS_is_win),$(WORKDIR.lib)/$(core_y:%.$(MAJORBINARY).dll=%_dll.lib))
$(WORKDIR.lib)/$(parameters_y.dpc): LOPT += $(if $(OS_is_win), $(if $(libsycl),$(libsycl),$(libsycl.default)) OpenCL.lib)
ifdef OS_is_win
$(WORKDIR.lib)/$(parameters_y.dpc:%.$(MAJORBINARY).dll=%_dll.lib): $(WORKDIR.lib)/$(parameters_y.dpc)
endif
endif

$(ONEAPI.tmpdir_y)/dll.res: $(VERSION_DATA_FILE)
$(ONEAPI.tmpdir_y)/dll.res: RCOPT += $(addprefix -I, $(WORKDIR) $(CORE.SERV.srcdir))
$(ONEAPI.tmpdir_y)/dll.res: $(CPPDIR.onedal)/onedal_dll.rc | $(ONEAPI.tmpdir_y)/. ; $(RC.COMPILE)

$(ONEAPI.tmpdir_y.dpc)/dll.res: $(VERSION_DATA_FILE)
$(ONEAPI.tmpdir_y.dpc)/dll.res: RCOPT += $(addprefix -I, $(WORKDIR) $(CORE.SERV.srcdir)) \
                                         -DONEDAL_DLL_RC_DATA_PARALLEL
$(ONEAPI.tmpdir_y.dpc)/dll.res: $(CPPDIR.onedal)/onedal_dll.rc | $(ONEAPI.tmpdir_y)/. ; $(RC.COMPILE)

#===============================================================================
# Threading parts
#===============================================================================
THR.srcs     := threading.cpp service_thread_pinner.cpp
THR.tmpdir_a := $(WORKDIR)/threading_static
THR.tmpdir_y := $(WORKDIR)/threading_dynamic
THR_TBB.objs_a := $(addprefix $(THR.tmpdir_a)/,$(THR.srcs:%.cpp=%_tbb.$o))
THR_TBB.objs_y := $(addprefix $(THR.tmpdir_y)/,$(THR.srcs:%.cpp=%_tbb.$o))

-include $(THR.tmpdir_a)/*.d
-include $(THR.tmpdir_y)/*.d

$(WORKDIR.lib)/$(thr_tbb_a): LOPT:=
$(WORKDIR.lib)/$(thr_tbb_a): $(THR_TBB.objs_a) ; $(LINK.STATIC)

$(WORKDIR.lib)/$(thr_tbb_y): LOPT += $(-fPIC) $(daaldep.rt.thr)
$(WORKDIR.lib)/$(thr_tbb_y): LOPT += $(if $(OS_is_win),-IMPLIB:$(@:%.dll=%_dll.lib),)
$(WORKDIR.lib)/$(thr_tbb_y): $(THR_TBB.objs_y) $(if $(OS_is_win),$(THR.tmpdir_y)/dll_tbb.res,) ; $(LINK.DYNAMIC) ; $(LINK.DYNAMIC.POST)

THR.objs_a := $(THR_TBB.objs_a)
THR.objs_y := $(THR_TBB.objs_y)
THR_TBB.objs := $(THR_TBB.objs_a) $(THR_TBB.objs_y)
THR.objs := $(THR.objs_a) $(THR.objs_y)

$(THR.objs): COPT += $(-fPIC) $(-cxx17) $(-Zl) $(-DEBC) -DDAAL_HIDE_DEPRECATED -DTBB_USE_ASSERT=0 -D_ENABLE_ATOMIC_ALIGNMENT_FIX

$(THR.objs_a): $(THR.tmpdir_a)/thr_inc_a_folders.txt
$(THR.objs_a): COPT += @$(THR.tmpdir_a)/thr_inc_a_folders.txt

$(THR.objs_y): $(THR.tmpdir_y)/thr_inc_y_folders.txt
$(THR.objs_y): COPT += @$(THR.tmpdir_y)/thr_inc_y_folders.txt
$(THR.objs_y): COPT += -D__DAAL_IMPLEMENTATION

$(THR.tmpdir_a)/thr_inc_a_folders.txt: makefile.lst | $(THR.tmpdir_a)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))
$(THR.tmpdir_y)/thr_inc_y_folders.txt: makefile.lst | $(THR.tmpdir_y)/. $(CORE.incdirs) ; $(call WRITE.PREREQS,$(addprefix -I, $(CORE.incdirs)),$(space))

$(THR_TBB.objs_a): $(THR.tmpdir_a)/%_tbb.$o: $(THR.srcdir)/%.cpp | $(THR.tmpdir_a)/. ; $(C.COMPILE)
$(THR_TBB.objs_y): $(THR.tmpdir_y)/%_tbb.$o: $(THR.srcdir)/%.cpp | $(THR.tmpdir_y)/. ; $(C.COMPILE)

$(THR.tmpdir_y)/dll_tbb.res: $(VERSION_DATA_FILE)
$(THR.tmpdir_y)/dll_tbb.res: RCOPT += -D_DAAL_THR_TBB $(addprefix -I, $(CORE.incdirs.common))

$(THR.tmpdir_y)/%_tbb.res: %.rc | $(THR.tmpdir_y)/. ; $(RC.COMPILE)

#===============================================================================
# Top level targets
#===============================================================================
daal: $(if $(CORE.ALGORITHMS.CUSTOM),           \
          _daal _release_c,                     \
          _daal _release _release_doc           \
      )
daal_c: _daal _release_c

oneapi: oneapi_c oneapi_dpc
oneapi_c: _oneapi_c _release_oneapi_c _release_parameters_c
oneapi_dpc: _oneapi_dpc _release_oneapi_dpc _release_parameters_dpc

onedal: oneapi daal
onedal_c: daal_c oneapi_c
onedal_dpc: daal_c oneapi_c oneapi_dpc

_daal:    _daal_core _daal_thr

_daal_core:  info.building.core
_daal_core:  $(WORKDIR.lib)/$(core_a) $(WORKDIR.lib)/$(core_y) ## TODO: move list of needed libs to one env var!!!
_daal_thr:   info.building.threading
_daal_thr:   $(if $(DAALTHRS),$(foreach ithr,$(DAALTHRS),_daal_thr_$(ithr)),)
_daal_thr_tbb:   $(WORKDIR.lib)/$(thr_tbb_a) $(WORKDIR.lib)/$(thr_tbb_y)

_release:    _release_c
_release_c:  _release_c_h _release_common

ifeq ($(BUILD_PARAMETERS_LIB),yes)
_parameters_c: info.building.parameters.C++.part
_parameters_c: $(WORKDIR.lib)/$(parameters_a) $(WORKDIR.lib)/$(parameters_y)

_parameters_dpc: info.building.parameters.DPC++.part
_parameters_dpc: $(WORKDIR.lib)/$(parameters_a.dpc) $(WORKDIR.lib)/$(parameters_y.dpc)

_release_parameters_c: _parameters_c
_release_parameters_dpc: _parameters_dpc
else
_release_parameters_c:
_release_parameters_dpc:
endif

_oneapi_c: info.building.oneapi.C++.part
_oneapi_c: $(WORKDIR.lib)/$(oneapi_a) $(WORKDIR.lib)/$(oneapi_y)

_oneapi_dpc: info.building.oneapi.DPC++.part
_oneapi_dpc: $(WORKDIR.lib)/$(oneapi_a.dpc) $(WORKDIR.lib)/$(oneapi_y.dpc)

_release_oneapi_c: _release_oneapi_c_h _release_oneapi_common
_release_oneapi_dpc: _release_oneapi_c _release_oneapi_common

#-------------------------------------------------------------------------------
# Populating RELEASEDIR
#-------------------------------------------------------------------------------
upd = $(cpy)

_release: info.building.release

#----- releasing static and dynamic libraries
define .release.y_win
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/.
	cp -fp $(WORKDIR.lib)/$1 $2/$1
endef
define .release.a_win
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/.
	cp -fp $(WORKDIR.lib)/$1 $2/$1
ifneq (,$(findstring dll.,$1))
	cp -fp $(WORKDIR.lib)/$1 $2/$(subst dll.,dll.$(MAJORBINARY).,$1)
endif
endef
define .release.y_link
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/.
	cp -fp $(WORKDIR.lib)/$1 $2/$(subst .$y,$(y_full_name_postfix),$1) && cd $2 && ln -sf $(subst .$y,$(y_full_name_postfix),$1) $(subst .$y,$(y_major_name_postfix),$1) && ln -sf $(subst .$y,$(y_major_name_postfix),$1) $1
endef
define .release.a
$3: $2/$1
$(if $(phony-upd),$(eval .PHONY: $2/$1))
$2/$1: $(WORKDIR.lib)/$1 | $2/. ; $(value upd)
endef

ifeq ($(if $(or $(OS_is_lnx),$(OS_is_mac)),yes,),yes)
$(foreach x,$(release.LIBS_A),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_c)))
$(foreach x,$(release.LIBS_Y),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_c)))
$(foreach x,$(release.ONEAPI.LIBS_A),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_Y),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_A.dpc),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_oneapi_dpc)))
$(foreach x,$(release.ONEAPI.LIBS_Y.dpc),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_oneapi_dpc)))

ifeq ($(BUILD_PARAMETERS_LIB),yes)
$(foreach x,$(release.PARAMETERS.LIBS_A),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_parameters_c)))
$(foreach x,$(release.PARAMETERS.LIBS_Y),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_parameters_c)))
$(foreach x,$(release.PARAMETERS.LIBS_A.dpc),$(eval $(call .release.a,$x,$(RELEASEDIR.libia),_release_parameters_dpc)))
$(foreach x,$(release.PARAMETERS.LIBS_Y.dpc),$(eval $(call .release.y_link,$x,$(RELEASEDIR.soia),_release_parameters_dpc)))
endif
endif

ifeq ($(OS_is_win),yes)
$(foreach x,$(release.LIBS_A),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_c)))
$(foreach x,$(release.LIBS_Y),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_c)))
$(foreach x,$(release.ONEAPI.LIBS_A),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_Y),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.LIBS_A.dpc),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_oneapi_dpc)))
$(foreach x,$(release.ONEAPI.LIBS_Y.dpc),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_oneapi_dpc)))

ifeq ($(BUILD_PARAMETERS_LIB),yes)
$(foreach x,$(release.PARAMETERS.LIBS_A),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_parameters_c)))
$(foreach x,$(release.PARAMETERS.LIBS_Y),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_parameters_c)))
$(foreach x,$(release.PARAMETERS.LIBS_A.dpc),$(eval $(call .release.a_win,$x,$(RELEASEDIR.libia),_release_parameters_dpc)))
$(foreach x,$(release.PARAMETERS.LIBS_Y.dpc),$(eval $(call .release.y_win,$x,$(RELEASEDIR.soia),_release_parameters_dpc)))
endif
endif

_release_c: ./deploy/pkg-config/pkg-config.tpl
	python ./deploy/pkg-config/generate_pkgconfig.py --output_dir $(RELEASEDIR.pkgconfig) --template_name ./deploy/pkg-config/pkg-config.tpl

#----- releasing examples
define .release.x
$3: $2/$(subst _$(_OS),,$1)
$2/$(subst _$(_OS),,$1): $(DIR)/$1 | $(dir $2/$1)/.
	$(if $(filter %makefile_win,$1),python ./deploy/local/generate_win_makefile.py $(dir $(DIR)/$1) $(dir $2/$1),$(value cpy))
	$(if $(filter %.sh %.bat,$1),chmod +x $$@)
endef
$(foreach x,$(release.EXAMPLES.DATA),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_common)))
$(foreach x,$(release.EXAMPLES.CPP),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_c)))
$(foreach x,$(release.ONEAPI.EXAMPLES.CPP),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_oneapi_c)))
$(foreach x,$(release.ONEAPI.EXAMPLES.DPC),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_oneapi_dpc)))
$(foreach x,$(release.ONEAPI.EXAMPLES.DATA),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_oneapi_common)))
$(foreach x,$(release.EXAMPLES.CMAKE),$(eval $(call .release.x,$x,$(RELEASEDIR.daal),_release_common)))

#----- releasing environment scripts
define .release.x
$4: $3/$2
$3/$2: $(DIR)/$1 | $3/. ; $(value cpy)
	$(if $(filter %.sh %.bat dal,$2),sed -i -e 's/__DAL_MAJOR_BINARY__/$(MAJORBINARY)/' $3/$2)
	$(if $(filter %.sh %.bat dal,$2),sed -i -e 's/__DAL_MINOR_BINARY__/$(MINORBINARY)/' $3/$2)
	$(if $(OS_is_win),unix2dos $3/$2)
	$(if $(filter %.sh %.bat,$2),chmod +x $$@)
endef
$(foreach x,$(release.ENV),$(eval $(call .release.x,$x,$(notdir $(subst _$(_OS),,$x)),$(RELEASEDIR.env),_release_common)))
$(if $(OS_is_lnx),$(foreach x,$(release.MODULEFILES),$(eval $(call .release.x,$x,$(notdir $x),$(RELEASEDIR.modulefiles),_release_common))))
$(foreach x,$(release.CONF),$(eval $(call .release.x,$x,$(notdir $(subst _$(_OS),,$x)),$(RELEASEDIR.conf),_release_common)))

#----- releasing documentation
_release_doc:
define .release.d
_release_doc: $2
$2: $1 | $(dir $2)/. ; $(value cpy)
	$(if $(filter %.sh %.bat,$2),chmod +x $$@)
endef
$(foreach d,$(release.DOC.COMMON),    $(eval $(call .release.d,$d,$(subst $(DOC.srcdir),    $(RELEASEDIR.doc),    $(subst _$(_OS),,$d)))))
$(foreach d,$(release.DOC.OSSPEC),    $(eval $(call .release.d,$d,$(subst $(DOC.srcdir),    $(RELEASEDIR.doc),    $(subst _$(_OS),,$d)))))

#----- releasing samples and headers
define .release.d
$3: $2
$2: $1 | $(dir $2)/. ; $(value cpy)
	$(if $(filter %.sh %.bat,$2),chmod +x $$@)
endef
$(foreach d,$(release.SAMPLES.CPP),   $(eval $(call .release.d,$d,$(subst $(SAMPLES.srcdir),$(RELEASEDIR.samples),$(subst _$(_OS),,$d)),_release_c)))
$(foreach d,$(release.SAMPLES.ONEDAL.DPC),   $(eval $(call .release.d,$d,$(subst $(SAMPLES.srcdir),$(RELEASEDIR.samples),$(subst _$(_OS),,$d)),_release_oneapi_dpc)))
$(foreach d,$(release.SAMPLES.CMAKE),   $(eval $(call .release.d,$d,$(subst $(SAMPLES.srcdir),$(RELEASEDIR.samples),$d),_release_common)))

$(CORE.incdirs): _release_c_h

define .release.dd
$3: $2
$2: $1 ; $(value mkdir)$(value cpy)
	$(if $(filter %library_version_info.h,$2),+$(daalmake) -f makefile update_headers_version)
	$(if $(USECPUS.out.defs.filter),$(if $(filter %daal_kernel_defines.h,$2),$(USECPUS.out.defs.filter) $2; rm -rf $(subst .h,.h.bak,$2)))
endef
$(foreach d,$(release.HEADERS.COMMON),$(eval $(call .release.dd,$d,$(subst $(CPPDIR.daal)/include/,$(RELEASEDIR.include)/,$d),_release_c_h)))
$(foreach d,$(release.HEADERS.OSSPEC),$(eval $(call .release.dd,$d,$(subst $(CPPDIR.daal)/include/,$(RELEASEDIR.include)/,$(subst _$(_OS),,$d)),_release_c_h)))

define .release.oneapi.dd
$3: $2
$2: $1 ; $(value mkdir)$(value cpy)
endef
$(foreach d,$(release.ONEAPI.HEADERS.COMMON),$(eval $(call .release.oneapi.dd,$d,$(subst $(CPPDIR)/,$(RELEASEDIR.include)/,$d),_release_oneapi_c_h)))
$(foreach d,$(release.ONEAPI.HEADERS.OSSPEC),$(eval $(call .release.oneapi.dd,$d,$(subst $(CPPDIR)/,$(RELEASEDIR.include)/,$(subst _$(_OS),,$d)),_release_oneapi_c_h)))

#----- releasing static/dynamic Intel(R) TBB libraries
$(RELEASEDIR.tbb.libia) $(RELEASEDIR.tbb.soia): _release_common

define .release.t
_release_common: $2/$(notdir $1)
$2/$(notdir $1): $(call frompf1,$1) | $2/. ; $(value cpy)
endef
$(foreach t,$(releasetbb.LIBS_Y),$(eval $(call .release.t,$t,$(RELEASEDIR.tbb.soia))))
$(foreach t,$(releasetbb.LIBS_A),$(eval $(call .release.t,$t,$(RELEASEDIR.tbb.libia))))

#----- cmake configs generation

_release_cmake_configs:
	$(if $(shell bash -c "command -v cmake"),cmake -DINSTALL_DIR=$(RELEASEDIR.lib)/cmake/oneDAL -DARCH_DIR_ONEDAL=$(ARCH_DIR_ONEDAL) -P cmake/scripts/generate_config.cmake,echo 'cmake configs generation skipped')

#----- nuspecs generation
_release_common: _release_nuspec
_release_nuspec: update_headers_version _release_cmake_configs
	mkdir -p $(RELEASEDIR.nuspec)
	bash ./deploy/nuget/prepare_dal_nuget.sh --release-dir $(RELEASEDIR)

#===============================================================================
# Miscellaneous stuff
#===============================================================================

.PHONY: clean cleanrel cleanall
clean:    ; -rm -rf $(WORKDIR)
cleanrel: ; -rm -rf $(RELEASEDIR)
cleanall: clean cleanrel

define help
Usage: make [target...] [flag=value...]
Targets:
  daal      - build all (use -j to speedup the build)
  _daal_core ... - build only a part of the product,
             without populating release directory (read makefile for details)
  _release - populate release directory
  clean    - clean working directory $(WORKDIR)
  cleanrel - clean release directory $(RELEASEDIR) (for entire OS!)
  cleanall - clean both working and release directories
Flags:
  COMPILER   - compiler to use ($(COMPILERs)) [default: $(COMPILER)]
  WORKDIR    - directory for intermediate results [default: $(WORKDIR)]
  RELEASEDIR - directory for release [default: $(RELEASEDIR)]
  CORE.ALGORITHMS.CUSTOM - list of algorithms to be included into library
      build cpp interfaces only
      possible values: $(CORE.ALGORITHMS.CUSTOM.AVAILABLE)
  REQCPU - list of CPU optimizations to be included into library
      possible values: $(CPUs)
  REQDBG - flag that enables build in debug mode
  REQPROFILE - flag that enables kernel profiling using <ittnotify.h>
endef

daal_dbg:
	@echo "1" "!$(MKLDIR)!"
