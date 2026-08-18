# file: clang.ref.arm.mk
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

#++
#  Clang definitions for makefile
#--

include dev/make/compiler_definitions/clang.mk

PLATs.clang = lnxarm winarm

ifneq ($(SVE_SUPPORTED),1)
    $(error SVE is not supported on this system — aborting build. Pass SVE_SUPPORTED=1 to override.)
endif

LINKERS_SUPPORTED := bfd gold lld

ifneq ($(LINKER),)
    ifneq ($(filter $(LINKER),bfd gold lld),$(LINKER))
        $(error Invalid LINKER '$(LINKER)'. Supported on Linux: bfd gold lld)
    endif
endif

COMPILER.lnx.clang.target = $(if $(filter yes,$(COMPILER_is_cross)),--target=aarch64-linux-gnu)
COMPILER.win.clang.target = --target=aarch64-pc-windows-msvc

COMPILER.sysroot = $(if $(SYSROOT),--sysroot $(SYSROOT))

COMPILER.lnx.clang= clang++ -march=armv8-a+sve \
                     -DDAAL_REF -DONEDAL_REF -DDAAL_CPU=sve -Werror -Wno-empty-body -Wreturn-type \
                     $(COMPILER.lnx.clang.target) \
                     $(COMPILER.sysroot)
COMPILER.win.clang= clang-cl -march=armv8-a+sve \
                     -fms-runtime-lib=$(if $(MSVC_RT_is_release),dll,dll_dbg) \
                     -DDAAL_REF -DONEDAL_REF -DDAAL_CPU=sve -Werror -Wno-empty-body -Wreturn-type -Wno-deprecated-declarations \
                     $(COMPILER.win.clang.target)

# Linker flags
linker.ld.flag := $(if $(LINKER),-fuse-ld=$(LINKER),)
link.dynamic.lnx.clang = clang++ -march=armv8-a+sve \
                         $(linker.ld.flag) \
                         $(COMPILER.lnx.clang.target) \
                         $(COMPILER.sysroot)
link.dynamic.win.clang = lld-link
link.static.win.clang = llvm-lib

pedantic.opts.lnx.clang = $(pedantic.opts.clang)

# For SVE
a8sve_OPT.clang = $(-Q)march=armv8-a+sve
