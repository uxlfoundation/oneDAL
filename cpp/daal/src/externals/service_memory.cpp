/* file: service_memory.cpp */
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
//  Implementation of memory service functions
//--
*/

#include "src/externals/service_memory.h"
#include "src/externals/service_service.h"
#include "src/services/service_profiler.h"

void * daal::services::daal_malloc(size_t size, size_t alignment)
{
    // DAAL_PROFILER_SERVICE_TASK(daal_malloc);
    return daal::internal::ServiceInst::serv_malloc(size, alignment);
}

void * daal::services::daal_calloc(size_t size, size_t alignment)
{
    void * ptr = daal::services::daal_malloc(size, alignment);
    if (ptr == NULL)
    {
        return NULL;
    }

    char * cptr = (char *)ptr;

    daal::services::internal::service_memset_seq<char, DAAL_BASE_CPU>(cptr, '\0', size);

    return ptr;
}

void daal::services::daal_free(void * ptr)
{
    daal::internal::ServiceInst::serv_free(ptr);
}

void daal::services::daal_memmove_s(void * dest, size_t destSize, const void * src, size_t smax)
{
    daal::internal::ServiceInst::serv_memmove_s(dest, destSize, src, smax);
}

namespace daal
{
namespace services
{
void daal_free_buffers()
{
    daal::internal::ServiceInst::serv_free_buffers();
}
} // namespace services
} // namespace daal

void daal::services::daal_memcpy_s(void * dest, size_t destSize, const void * src, size_t srcSize)
{
    DAAL_PROFILER_SERVICE_TASK(daal_memcpy_s);
    size_t copySize = srcSize;
    if (destSize < srcSize)
    {
        copySize = destSize;
    }

    size_t BLOCKSIZE       = 200000000; // approx 200MB
    size_t nBlocks         = copySize / BLOCKSIZE;
    size_t sizeOfLastBlock = 0;
    if (nBlocks * BLOCKSIZE != copySize)
    {
        sizeOfLastBlock = copySize - (nBlocks * BLOCKSIZE);
    }

    char * dstChar = (char *)dest;
    char * srcChar = (char *)src;
    for (size_t i = 0; i < nBlocks; i++)
    {
        daal::internal::ServiceInst::serv_memcpy_s(&dstChar[i * BLOCKSIZE], BLOCKSIZE, &srcChar[i * BLOCKSIZE], BLOCKSIZE);
    }
    if (sizeOfLastBlock != 0)
    {
        daal::internal::ServiceInst::serv_memcpy_s(&dstChar[nBlocks * BLOCKSIZE], sizeOfLastBlock, &srcChar[nBlocks * BLOCKSIZE], sizeOfLastBlock);
    }
}

int daal::services::internal::daal_memcpy_s(void * dest, size_t destSize, const void * src, size_t srcSize)
{
    size_t copySize = srcSize;
    if (destSize < srcSize)
    {
        copySize = destSize;
    }

    size_t BLOCKSIZE       = 200000000; // approx 200MB
    size_t nBlocks         = copySize / BLOCKSIZE;
    size_t sizeOfLastBlock = 0;
    int result             = 0;
    if (nBlocks * BLOCKSIZE != copySize)
    {
        sizeOfLastBlock = copySize - (nBlocks * BLOCKSIZE);
    }

    char * dstChar = (char *)dest;
    char * srcChar = (char *)src;
    for (size_t i = 0; i < nBlocks; i++)
    {
        result |= daal::internal::ServiceInst::serv_memcpy_s(&dstChar[i * BLOCKSIZE], BLOCKSIZE, &srcChar[i * BLOCKSIZE], BLOCKSIZE);
    }
    if (sizeOfLastBlock != 0)
    {
        result |= daal::internal::ServiceInst::serv_memcpy_s(&dstChar[nBlocks * BLOCKSIZE], sizeOfLastBlock, &srcChar[nBlocks * BLOCKSIZE],
                                                             sizeOfLastBlock);
    }
    return result;
}
