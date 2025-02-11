#===============================================================================
# Copyright 2023 Intel Corporation
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
#  Math backend (OpenBLAS) definitions for makefile
#--

OPENBLASDIR:= $(if $(wildcard $(DIR)/__deps/open_blas/*),$(DIR)/__deps/open_blas,                            \
                $(if $(wildcard $(OPENBLASROOT)/include/*),$(subst \,/,$(OPENBLASROOT)),                              \
                    $(error Can`t find OPENBLAS libs in $(DIR)/__deps/open_blas or OPENBLASROOT.)))
OPENBLASDIR.include := $(OPENBLASDIR)/include
OPENBLASDIR.libia := $(OPENBLASDIR)/lib

daaldep.math_backend.thr := $(OPENBLASDIR.libia)/libopenblas.$a
daaldep.math_backend.seq := $(OPENBLASDIR.libia)/libopenblas.$a

daaldep.math_backend.incdir := $(OPENBLASDIR.include)
daaldep.math_backend_oneapi.incdir := $(OPENBLASDIR.include)

ifeq ($(RNG_OPENRNG), yes)
	OPENRNGDIR:= $(if $(wildcard $(DIR)/__deps/openrng/*),$(DIR)/__deps/openrng,                            \
					$(if $(wildcard $(OPENRNGROOT)/include/*),$(subst \,/,$(OPENRNGROOT)),                              \
						$(error Can`t find OPENRNG libs in $(DIR)/__deps/openrng or OPENRNGROOT)))
	OPENRNGDIR.include := $(OPENRNGDIR)/include
	OPENRNGDIR.libia := $(OPENRNGDIR)/lib

	daaldep.rng_backend.incdir := $(OPENRNGDIR.include)
	daaldep.rng_backend.lib := $(OPENRNGDIR.libia)/libopenrng.$a

	daaldep.math_backend.incdir += $(daaldep.rng_backend.incdir)
endif

daaldep.math_backend.ext := $(daaldep.math_backend.thr)
daaldep.math_backend.sycl := $(daaldep.math_backend.thr)
daaldep.math_backend.oneapi := $(daaldep.math_backend.thr)
