#===============================================================================
# Copyright 2014 Intel Corporation
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

ifeq ($(PLAT),)
    PLAT:=$(shell bash dev/make/identify_os.sh)
endif

ifeq (help,$(MAKECMDGOALS))
    PLAT:=win32e
endif

TARGETs = x86_64
TARGET ?= x86_64

$(if $(filter $(TARGETs),$(TARGET)),,$(error TARGET must be one of $(TARGETs)))

include dev/make/$(TARGET)/makefile.mk
