/* file: service_service_mkl.h */
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
//  Declaration of math service functions
//--
*/

#ifndef __SERVICE_SERVICE_MKL_H__
#define __SERVICE_SERVICE_MKL_H__

#include "services/daal_defines.h"
#include "src/services/service_topo.h"
#include <mkl.h>
#include <mkl_service.h>
#include <string.h>
#ifdef USE_STD_ALLOC
    #include <cstdlib>
#endif

namespace daal
{
namespace internal
{
namespace mkl
{
struct MklService
{
    static void * serv_malloc(size_t size, size_t alignment)
    {
#ifndef USE_STD_ALLOC
        return MKL_malloc(size, alignment);
#else
        if (size < alignment) size = alignment;
        const size_t mod = size % alignment;
        if (mod) size += alignment - mod;
        return std::aligned_alloc(alignment, size);
#endif
    }

    static void serv_free(void * ptr)
    {
#ifndef USE_STD_ALLOC
        MKL_free(ptr);
#else
        std::free(ptr);
#endif
    }

    static void serv_free_buffers() { MKL_Free_Buffers(); }

    static int serv_memcpy_s(void * dest, size_t destSize, const void * src, size_t srcSize)
    {
        if (destSize < srcSize) return static_cast<int>(ENOMEM);
        memcpy(dest, src, srcSize);
        return 0;
        // TODO: safe funtion
        // return memcpy_s(dest, destSize, src, srcSize);
    }

    static int serv_memmove_s(void * dest, size_t destSize, const void * src, size_t smax)
    {
        if (destSize < smax) return static_cast<int>(ENOMEM);
        memmove(dest, src, smax);
        return 0;
        // TODO: safe funtion
        // return memmove_s(dest, destSize, src, smax);
    }

    static int serv_get_ht()
    {
        const int ncorespercpu = serv_get_ncorespercpu();
        if (ncorespercpu < 0)
        {
            // Failed to get the number of cores per CPU.
            return -1;
        }
        return (ncorespercpu > 1 ? 1 : 0);
    }

    static int serv_get_ncpus()
    {
        if (daal::services::internal::_internal_daal_GetStatus() != 0)
        {
            // CPU topology initialization failed;
            return -1;
        }
        const unsigned int ncores = daal::services::internal::_internal_daal_GetProcessorCoreCount();
        return (ncores ? ncores : 1);
    }

    static int serv_get_ncorespercpu()
    {
        if (daal::services::internal::_internal_daal_GetStatus() != 0)
        {
            // CPU topology initialization failed;
            return -1;
        }
        const unsigned int nlogicalcpu = daal::services::internal::_internal_daal_GetSysLogicalProcessorCount();
        const unsigned int ncpus       = serv_get_ncpus();
        // On hybrid systems, return 2 in case of hyper-threading enabled at least on some cores,
        // 1 otherwise
        return (ncpus > 0 && nlogicalcpu > 0 && nlogicalcpu > ncpus ? (nlogicalcpu - ncpus + 1) / ncpus : 1);
    }

    // TODO: The real call should be delegated to a backend library if the option is supported
    static int serv_set_memory_limit(int type, size_t limit) { return MKL_Set_Memory_Limit(type, limit); }
    // Added for interface compatibility - not expected to be called
    static size_t serv_strnlen_s(const char * src, size_t slen) { return strnlen(src, slen); }

    static int serv_strncpy_s(char * dest, size_t dmax, const char * src, size_t slen)
    {
        if (dmax < slen) return static_cast<int>(ENOMEM);
        strncpy(dest, src, slen);
        return 0;
    }

    static int serv_strncat_s(char * dest, size_t dmax, const char * src, size_t slen)
    {
        if (dmax < slen) return static_cast<int>(ENOMEM);
        strncat(dest, src, slen);
        return 0;
    }

    // TODO: not a safe function - no control for the input buffer end
    static double serv_string_to_double(const char * nptr, char ** endptr)
    {
        const char * cur = nptr;
        for (; isdigit(*cur) || *cur == '-' || *cur == 'e' || *cur == 'E' || *cur == '.'; ++cur)
            ;
        if (endptr) *endptr = const_cast<char *>(cur);
        size_t size = cur - nptr;
        // TODO replace with static buffer
        char * buffer = static_cast<char *>(malloc(size + 1));
        for (size_t i = 0; i < size; ++i) buffer[i] = nptr[i];
        buffer[size] = '\0';
        double val   = atof(buffer);
        free(buffer);
        return val;
    }

    static float serv_string_to_float(const char * nptr, char ** endptr) { return static_cast<float>(serv_string_to_double(nptr, endptr)); }

    // TODO: not a safe function - no control for the input buffer end
    static int serv_string_to_int(const char * nptr, char ** endptr)
    {
        const char * cur = nptr;
        for (; isdigit(*cur) || *cur == '-'; ++cur)
            ;
        if (endptr) *endptr = const_cast<char *>(cur);
        size_t size = cur - nptr;
        // TODO replace with static buffer
        char * buffer = static_cast<char *>(malloc(size + 1));
        for (size_t i = 0; i < size; ++i) buffer[i] = nptr[i];
        buffer[size] = '\0';
        int val      = atoi(buffer);
        free(buffer);
        return val;
    }

    static int serv_int_to_string(char * buffer, size_t n, int value) { return snprintf(buffer, n, "%d", value); }

    static int serv_double_to_string(char * buffer, size_t n, double value) { return snprintf(buffer, n, "%E", value); }
};

} // namespace mkl
} // namespace internal
} // namespace daal

#endif
