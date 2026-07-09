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

MKLGPUDIR:= $(subst \,/,$(MKLROOT))
MKLGPUDIR.include := $(MKLGPUDIR)/include/oneapi
MKLGPUDIR.lib   := $(MKLGPUDIR)/lib

mklgpu.HEADERS := $(MKLGPUDIR.include)/mkl.hpp

daaldep.math_backend.incdir := $(MKLDIR.include)
daaldep.math_backend_oneapi.incdir := $(MKLDIR.include) $(MKLGPUDIR.include)

daaldep.lnx32e.mkl.thr := $(MKLDIR.libia)/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.core := $(MKLDIR.libia)/$(plib)mkl_core.$a 
daaldep.lnx32e.mkl.interfaces := $(MKLDIR.libia)/$(plib)mkl_intel_ilp64.$a
daaldep.lnx32e.mkl.sycl_thr := $(MKLDIR.libia)/$(plib)mkl_gnu_thread.$(so)
daaldep.lnx32e.mkl.sycl_core := $(MKLDIR.libia)/$(plib)mkl_core.$(so)
daaldep.lnx32e.mkl.sycl_interfaces := $(MKLDIR.libia)/$(plib)mkl_intel_lp64.$(so)
daaldep.lnx32e.mkl.sycl_rng := $(MKLDIR.libia)/$(plib)mkl_sycl_rng.$(so)
daaldep.lnx32e.mkl.sycl_blas := $(MKLDIR.libia)/$(plib)mkl_sycl_blas.$(so)
daaldep.lnx32e.mkl.sycl_lapack := $(MKLDIR.libia)/$(plib)mkl_sycl_lapack.$(so)
daaldep.lnx32e.mkl.sycl_sparse := $(MKLDIR.libia)/$(plib)mkl_sycl_sparse.$(so)

# List of oneMKL libraries to exclude from linking.
# This list is used to generate the `--exclude-libs` linker options.
# If you need to exclude additional libraries, extend this list by appending the library names.
MATH_LIBS_TO_EXCLUDE := $(plib)mkl_tbb_thread.$a $(plib)mkl_core.$a $(plib)mkl_intel_ilp64.$a

daaldep.win32e.mkl.thr := $(MKLDIR.libia)/mkl_tbb_thread$d.$a
daaldep.win32e.mkl.core := $(MKLDIR.libia)/mkl_core.$a
daaldep.win32e.mkl.interfaces := $(MKLDIR.libia)/mkl_intel_ilp64.$a
daaldep.win32e.mkl.sycl_thr := $(MKLDIR.libia)/mkl_intel_thread_dll.$a
daaldep.win32e.mkl.sycl_core := $(MKLDIR.libia)/mkl_core_dll.$a
daaldep.win32e.mkl.sycl_interfaces := $(MKLDIR.libia)/mkl_intel_lp64_dll.$a
daaldep.win32e.mkl.sycl_rng := $(MKLDIR.libia)/mkl_sycl_rng_dll.$a
daaldep.win32e.mkl.sycl_blas := $(MKLDIR.libia)/mkl_sycl_blas_dll.$a
daaldep.win32e.mkl.sycl_lapack := $(MKLDIR.libia)/mkl_sycl_lapack_dll.$a
daaldep.win32e.mkl.sycl_sparse := $(MKLDIR.libia)/mkl_sycl_sparse_dll.$a

daaldep.fbsd32e.mkl.thr := $(MKLDIR.libia)/$(plib)mkl_tbb_thread.$a
daaldep.fbsd32e.mkl.interfaces := $(MKLDIR.libia)/$(plib)mkl_intel_ilp64.$a
daaldep.fbsd32e.mkl.core := $(MKLDIR.libia)/$(plib)mkl_core.$a

daaldep.math_backend.core     := $(daaldep.$(PLAT).mkl.core)
daaldep.math_backend.interfaces     := $(daaldep.$(PLAT).mkl.interfaces)
daaldep.math_backend.thr := $(daaldep.$(PLAT).mkl.thr)

daaldep.lnx32e.vml :=
daaldep.lnx32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.win32e.vml :=
daaldep.win32e.ipp :=

daaldep.mac32e.vml :=
daaldep.mac32e.ipp :=

daaldep.fbsd32e.vml :=
daaldep.fbsd32e.ipp := $(if $(COV.libia),$(COV.libia)/libcov.a)

daaldep.vml     := $(daaldep.$(PLAT).vml)
daaldep.ipp     := $(daaldep.$(PLAT).ipp)

# For the MKL-based math backend, we don't link MKL statically into libonedal_core library
# The user must provide MKL static libraries when building their application.
daaldep.math_backend.static_link_deps := 
# Static MKL libraries linked into the shared oneDAL library.
daaldep.math_backend.shared_link_deps := $(daaldep.ipp) $(daaldep.vml) $(daaldep.math_backend.interfaces) $(daaldep.math_backend.thr) $(daaldep.math_backend.core)
# Dynamic MKL libraries(SYCL) linked into the shared oneDAL(SYCL) library.
mkl_libs.lnx32e := -L$(MKLROOT)/lib \
    $(daaldep.lnx32e.mkl.sycl_blas) $(daaldep.lnx32e.mkl.sycl_lapack) $(daaldep.lnx32e.mkl.sycl_sparse) \
    $(daaldep.lnx32e.mkl.sycl_rng) \
    $(daaldep.lnx32e.mkl.sycl_core) $(daaldep.lnx32e.mkl.sycl_interfaces) $(daaldep.lnx32e.mkl.sycl_thr) \
    -lsycl -lm -ldl


mkl_libs.win32e :=  $(daaldep.win32e.mkl.sycl_blas) $(daaldep.win32e.mkl.sycl_lapack) $(daaldep.win32e.mkl.sycl_sparse) \
    $(daaldep.win32e.mkl.sycl_rng) \
    $(daaldep.win32e.mkl.sycl_core) $(daaldep.win32e.mkl.sycl_interfaces) $(daaldep.win32e.mkl.sycl_thr) libiomp5md.lib

mkl_libs.mac32e :=
mkl_libs.lnxarm :=
mkl_libs.lnxriscv64 :=

daaldep.math_backend.dpc_link_deps := $(mkl_libs.$(PLAT))
