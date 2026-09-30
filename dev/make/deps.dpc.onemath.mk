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
#  Device-side math backend (oneMath) definitions for makefile
#
#  oneMath (https://github.com/uxlfoundation/oneMath) implements the same DPC++
#  interface as oneMKL and additionally dispatches to cuBLAS, cuSOLVER, cuSPARSE
#  and cuRAND, which is what lets the unmodified oneDAL device sources run on
#  NVIDIA GPUs. Selected with DPC_MATH_BACKEND=onemath.
#
#  Only the device-side (`_dpc`) pieces are replaced here. The host math library
#  stays whatever BACKEND_CONFIG selected, so a build can take oneMKL on the CPU
#  and oneMath on the GPU.
#
#  Not every domain oneDAL uses is available in oneMath -- BLAS and LAPACK are,
#  RNG and sparse BLAS are not. See the "NVIDIA GPUs through oneMath" section of
#  dev/bazel/README.md for what that means in practice.
#--

ifeq (,$(ONEMATHROOT))
    $(error DPC_MATH_BACKEND=onemath requires ONEMATHROOT to point at a oneMath install)
endif

# There is no redistributable to download and no default install location: which
# CUDA backends a libonemath.so contains is fixed when oneMath itself is
# configured, so the library has to be built locally and named explicitly.
ONEMATHDIR := $(subst \,/,$(ONEMATHROOT))
ONEMATHDIR.include := $(ONEMATHDIR)/include
ONEMATHDIR.lib := $(ONEMATHDIR)/lib

ifeq (,$(wildcard $(ONEMATHDIR.include)/oneapi/math.hpp))
    $(error $(ONEMATHDIR.include)/oneapi/math.hpp not found; ONEMATHROOT=$(ONEMATHROOT) does not look like a oneMath install)
endif

# Windows and the non-x86 platforms are deliberately not wired up: the oneMath
# import-library naming and the NVPTX toolchain setup there are unverified, and a
# silently wrong link line is worse than a refusal.
ifneq ($(PLAT),lnx32e)
    $(error DPC_MATH_BACKEND=onemath is only supported for PLAT=lnx32e, got '$(PLAT)')
endif

# Appended rather than assigned: the oneAPI sources also include DAAL headers,
# which need the host backend's include directory whatever it is. Adding oneMath
# alongside is safe because <oneapi/math.hpp> and <oneapi/mkl.hpp> do not
# collide, and math_backend.hpp includes exactly one of them.
daaldep.math_backend_oneapi.incdir += $(ONEMATHDIR.include)

# Switches cpp/oneapi/dal/backend/math_backend.hpp from <oneapi/mkl.hpp> to
# <oneapi/math.hpp>. Defined next to the link line it belongs with, so the two
# cannot drift apart.
daaldep.math_backend.dpc_defines := -DONEDAL_MATH_BACKEND_ONEMATH

# Only the run-time dispatching library is linked. The per-domain backend shared
# objects that sit next to it (libonemath_blas_cublas.so and friends) are
# dlopened by the dispatcher, so they have to be findable at run time -- keep
# $(ONEMATHROOT)/lib on LD_LIBRARY_PATH -- but they are not link inputs, and
# which of them exist depends on how oneMath was configured.
onemath_libs.lnx32e := -L$(ONEMATHDIR.lib) -lonemath -lsycl -lm -ldl

daaldep.math_backend.dpc_link_deps := $(onemath_libs.$(PLAT))
