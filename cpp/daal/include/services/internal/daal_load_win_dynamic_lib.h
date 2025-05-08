/* file: daal_load_win_dynamic_lib.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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
//  Implementation of safe load library functionality for Windows.
//--
*/

#ifndef __DAAL_LOAD_WIN_DYNAMIC_LIB_H__
#define __DAAL_LOAD_WIN_DYNAMIC_LIB_H__

#if defined(_WIN32) || defined(_WIN64)

    #define NOMINMAX // Prevents Windows.h from defining min and max macros
    #include <windows.h>
    #include "services/daal_defines.h"

DAAL_EXPORT HMODULE _daal_load_win_dynamic_lib(LPCTSTR filename);

#endif // defined(_WIN32) || defined(_WIN64)

#endif // __DAAL_LOAD_WIN_DYNAMIC_LIB_H__
