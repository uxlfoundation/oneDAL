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

daaldep.lnx32e.mkl.thr := $(MKLDIR.libia)/$(plib)mkl_tbb_thread.$a
daaldep.lnx32e.mkl.seq := $(MKLDIR.libia)/$(plib)mkl_sequential.$a
daaldep.lnx32e.mkl.core := $(MKLDIR.libia)/$(plib)mkl_core.$a 
daaldep.lnx32e.mkl.interfaces := $(MKLDIR.libia)/$(plib)mkl_intel_ilp64.$a
daaldep.lnx32e.mkl.sycl := $(MKLGPUDIR.lib)/$(plib)mkl_sycl.$a

daaldep.win32e.mkl.thr := $(MKLDIR.libia)/mkl_tbb_thread$d.$a
daaldep.win32e.mkl.seq := $(MKLDIR.libia)/mkl_sequential.$a
daaldep.win32e.mkl.interfaces := $(MKLDIR.libia)/mkl_intel_ilp64.$a
daaldep.win32e.mkl.core := $(MKLDIR.libia)/mkl_core.$a
daaldep.win32e.mkl.sycl := $(MKLGPUDIR.lib)/mkl_sycl$d.$a

daaldep.fbsd32e.mkl.thr := $(MKLDIR.libia)/$(plib)mkl_tbb_thread.$a
daaldep.fbsd32e.mkl.seq := $(MKLDIR.libia)/$(plib)mkl_sequential.$a
daaldep.fbsd32e.mkl.interfaces := $(MKLDIR.libia)/$(plib)mkl_intel_ilp64.$a
daaldep.fbsd32e.mkl.core := $(MKLDIR.libia)/$(plib)mkl_core.$a
daaldep.fbsd32e.mkl.sycl := $(MKLGPUDIR.lib)/$(plib)mkl_sycl.$a

daaldep.math_backend.core     := $(daaldep.$(PLAT).mkl.core)
daaldep.math_backend.interfaces     := $(daaldep.$(PLAT).mkl.interfaces)
daaldep.math_backend.thr := $(daaldep.$(PLAT).mkl.thr)
daaldep.math_backend.seq := $(daaldep.$(PLAT).mkl.seq)
daaldep.math_backend.sycl := $(daaldep.$(PLAT).mkl.sycl)

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

daaldep.math_backend.ext := $(daaldep.ipp) $(daaldep.vml) $(daaldep.math_backend.interfaces) $(daaldep.math_backend.thr) $(daaldep.math_backend.core)
daaldep.math_backend.sycl := $(daaldep.math_backend.sycl)
