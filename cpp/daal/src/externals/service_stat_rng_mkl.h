/* file: service_stat_rng_mkl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Template wrappers for common Intel(R) MKL functions.
//--
*/

#ifndef __SERVICE_STAT_RNG_MKL_H__
#define __SERVICE_STAT_RNG_MKL_H__

#include <cstdint>

#define __DAAL_VSLFN_CALL_NR(f_name, f_args, errcode) __DAAL_VSLFN_CALL_NO_V(f_name, f_args, errcode)
/*
 Calls the MKL RNG functions in a loop while handling large n
 Each iteration handles a chunk of size less than 2^31 to avoid exceeding MKL's limits
 n  - 64-bit, the total number of random numbers to generate
 nn - 32-bit, the number of random numbers to generate in the current iteration
 rr - pointer to the array where the generated random numbers will be stored
 */
#define __DAAL_VSLFN_CALL_NR_WHILE(f_name, f_args, errcode)                 \
    {                                                                       \
        size_t nn_left = n;                                                 \
        while (nn_left > 0)                                                 \
        {                                                                   \
            nn = (nn_left > 0x7FFFFFFFL) ? 0x7FFFFFFF : (int32_t)(nn_left); \
                                                                            \
            __DAAL_VSLFN_CALL_V(f_name, f_args, errcode);                   \
            if (errcode < 0) return errcode;                                \
                                                                            \
            rr += nn;                                                       \
            nn_left -= nn;                                                  \
        }                                                                   \
    }

#define __DAAL_VSLFN_CALL_V(f_name, f_args, retcode) \
    {                                                \
        retcode = v##f_name f_args;                  \
    }

#define __DAAL_VSLFN_CALL_NO_V(f_name, f_args, retcode) \
    {                                                   \
        retcode = f_name f_args;                        \
    }

#endif
