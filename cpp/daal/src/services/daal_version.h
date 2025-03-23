/* file: daal_version.h */
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

#ifndef ___DAAL_VERSION_H__
#define ___DAAL_VERSION_H__

#include "daal_version_gen.h"

#if PRODUCT_STATUS == 'A'
    #define PRODUCTSTATUS          "alpha"
    #define FILE_VERSION0(a, b, c) #a "." #b "-alpha0" #c
    #define SUBBUILD               0
#endif
#if PRODUCT_STATUS == 'B'
    #define PRODUCTSTATUS          "beta"
    #define FILE_VERSION0(a, b, c) #a "." #b "-beta0" #c
    #define SUBBUILD               0
#endif
#if PRODUCT_STATUS == 'P'
    #define PRODUCTSTATUS          ""
    #define FILE_VERSION0(a, b, c) #a "." #b "." #c
    #define SUBBUILD               1
#endif

/* Intermediate defines */
#define FILE_VERSION1(a, b, c) FILE_VERSION0(a, b, c)

/* The next 3 defines need to use in *.rc files */
/* instead of symbolic constants like "10.0.2.0" */

#define FILE_VERSION MAJORVERSION, MINORVERSION, UPDATEVERSION, SUBBUILD

#define FILE_VERSION_STR    FILE_VERSION1(MAJORVERSION, MINORVERSION, UPDATEVERSION)
#define PRODUCT_VERSION_STR FILE_VERSION1(MAJORVERSION, MINORVERSION, UPDATEVERSION)

#define PRODUCT_NAME_STR "oneAPI Data Analytics Library\0"

#endif
