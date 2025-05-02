/* file: library_version_info.cpp */
/*******************************************************************************
* Copyright 2015 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Definitions of structures used for environment detection.
//--
*/

#include "services/library_version_info.h"
#include "src/services/daal_version.h"
#include "src/services/service_defines.h"
#include "services/env_detect.h"

#ifndef DAAL_REF // temporary!!! should depend both on BACKEND and TARGETARCH
    #include <mkl.h>
static const char * cpu_long_names[] = { "Generic",
                                         "Supplemental Streaming SIMD Extensions 3",
                                         "Intel(R) Streaming SIMD Extensions 4.2",
                                         "Intel(R) Advanced Vector Extensions",
                                         "Intel(R) Advanced Vector Extensions 2",
                                         "Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector Extensions 512",
                                         "Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512",
                                         "Intel(R) Xeon Phi(TM) processors based on Intel(R) Advanced Vector Extensions 512 with support of "
                                         "AVX512_4FMAPS and AVX512_4VNNIW instruction groups" };
#else
static const char * cpu_long_names[] = { "Generic" };
#endif

DAAL_EXPORT daal::services::LibraryVersionInfo::LibraryVersionInfo()
    : majorVersion(MAJORVERSION),
      minorVersion(MINORVERSION),
      updateVersion(UPDATEVERSION),
      productStatus(PRODUCTSTATUS),
      build(BUILD),
      build_rev(BUILD_REV),
      name(PRODUCT_NAME_STR),
#ifndef DAAL_REF
      processor(cpu_long_names[daal::services::Environment::getInstance()->getCpuId()])
#else
      processor(cpu_long_names[0])
#endif
{}

DAAL_EXPORT daal::services::LibraryVersionInfo::~LibraryVersionInfo() {}
