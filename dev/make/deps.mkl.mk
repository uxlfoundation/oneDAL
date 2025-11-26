#===============================================================================
# Copyright 2023 Intel Corporation
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
#  Math backend (MKL) definitions for makefile
#--

MKLDIR:= $(subst \,/,$(MKLROOT))
MKLDIR.include := $(MKLDIR)/include
MKLDIR.libia   := $(MKLDIR)/lib
RELEASEDIR.include.mklgpu := $(RELEASEDIR.include)/services/internal/sycl/math

MKLGPUDIR:= $(subst \,/,$(MKLROOT))
MKLGPUDIR.include := $(MKLGPUDIR)/include/oneapi
MKLGPUDIR.lib   := $(MKLGPUDIR)/lib

mklgpu.HEADERS := $(MKLGPUDIR.include)/mkl.hpp

daaldep.math_backend.incdir := $(MKLDIR.include)
daaldep.math_backend_oneapi.incdir := $(MKLDIR.include) $(MKLGPUDIR.include)

daaldep.lnx32e.mkl.thr_static := $(MKLDIR.libia)/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.thr_dynamic := $(MKLDIR.libia)/$(plib)mkl_tbb_thread.$(so)
daaldep.lnx32e.mkl.core_static := $(MKLDIR.libia)/$(plib)mkl_core.$a
daaldep.lnx32e.mkl.core_dynamic := $(MKLDIR.libia)/$(plib)mkl_core.$(so)
daaldep.lnx32e.mkl.interfaces_static := $(MKLDIR.libia)/$(plib)mkl_intel_ilp64.$a
daaldep.lnx32e.mkl.interfaces_dynamic := $(MKLDIR.libia)/$(plib)mkl_intel_ilp64.$(so)
daaldep.lnx32e.mkl.sycl := $(MKLGPUDIR.lib)/$(plib)mkl_sycl.$a
daaldep.lnx32e.mkl.sycl_dynamic := $(MKLGPUDIR.lib)/$(plib)mkl_sycl.$(so)

# List of oneMKL libraries to exclude from linking.
# This list is used to generate the `--exclude-libs` linker options.
# If you need to exclude additional libraries, extend this list by appending the library names.
MATH_LIBS_TO_EXCLUDE := $(plib)mkl_tbb_thread.$a $(plib)mkl_core.$a $(plib)mkl_intel_ilp64.$a $(plib)mkl_sycl.$a

daaldep.win32e.mkl.thr_dynamic := $(MKLDIR.libia)/mkl_tbb_thread$d.$(so)
daaldep.win32e.mkl.interfaces_dynamic := $(MKLDIR.libia)/mkl_intel_ilp64.$(so)
daaldep.win32e.mkl.core_dynamic := $(MKLDIR.libia)/mkl_core.$(so)
daaldep.win32e.mkl.sycl_dynamic := $(MKLGPUDIR.lib)/mkl_sycl$d.$(so)

daaldep.fbsd32e.mkl.thr := $(MKLDIR.libia)/$(plib)mkl_tbb_thread.$a
daaldep.fbsd32e.mkl.interfaces := $(MKLDIR.libia)/$(plib)mkl_intel_ilp64.$a
daaldep.fbsd32e.mkl.core := $(MKLDIR.libia)/$(plib)mkl_core.$a
daaldep.fbsd32e.mkl.sycl := $(MKLGPUDIR.lib)/$(plib)mkl_sycl.$a

daaldep.math_backend.core_static     := $(daaldep.$(PLAT).mkl.core_static)
daaldep.math_backend.core_dynamic     := $(daaldep.$(PLAT).mkl.core_dynamic)
daaldep.math_backend.interfaces_static     := $(daaldep.$(PLAT).mkl.interfaces_static)
daaldep.math_backend.interfaces_dynamic     := $(daaldep.$(PLAT).mkl.interfaces_dynamic)
daaldep.math_backend.thr_static := $(daaldep.$(PLAT).mkl.thr_static)
daaldep.math_backend.thr_dynamic := $(daaldep.$(PLAT).mkl.thr_dynamic)
daaldep.math_backend.sycl_dynamic := $(daaldep.$(PLAT).mkl.sycl_dynamic)


# For the MKL-based math backend, we don't link MKL statically into libonedal_core library
# The user must provide MKL static libraries when building their application.
daaldep.math_backend.static_link_deps := 
# Static MKL libraries linked into the shared oneDAL library.
daaldep.math_backend.shared_link_deps := $(daaldep.math_backend.interfaces_dynamic) $(daaldep.math_backend.thr_dynamic) $(daaldep.math_backend.core_dynamic)
# Static MKL libraries(SYCL) linked into the shared oneDAL(SYCL) library.
daaldep.math_backend.oneapi := $(daaldep.math_backend.sycl_dynamic)
