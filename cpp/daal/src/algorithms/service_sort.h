/* file: service_sort.h */
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
//  Implementation of sorting algorithms.
//--
*/

#ifndef __SERVICE_SORT_H__
#define __SERVICE_SORT_H__

#include "src/services/service_utils.h"
#include "src/algorithms/service_heap.h"
#include "src/threading/threading.h"
#include "services/collection.h"

#if defined(DAAL_INTEL_CPP_COMPILER)
    #include <immintrin.h>
#endif

namespace daal
{
namespace algorithms
{
namespace internal
{
using namespace daal::internal;
using namespace services::internal;

/**
 * \brief Quick sort function that sorts array x
 *
 * \param n[in]     Length of input arrays
 * \param x[in,out] Array to sort
 */
template <typename algorithmDataType, CpuType cpu>
void qSort(size_t n, algorithmDataType * x)
{
    daal::parallel_sort(x, x + n);
}

/**
 * \brief Sorts array x of daal::IdxValType elements by key; the order of equal keys is unspecified
 *
 * \param n[in]     Length of input array
 * \param x[in,out] Array to sort
 */
template <typename T, CpuType cpu>
void qSortByKey(size_t n, T * x)
{
    daal::parallel_sort_by_key(x, x + n);
}

/**
 *  \brief Quick sort function that sorts array x and rearranges array index
 *         accordingly
 *
 *  \param n[in] Length of input arrays
 *  \param x     Array that is used as "key" when sorted
 *  \param index Array that is used as "value" when sorted
 */
template <typename algorithmDataType, typename algorithmIndexType, CpuType cpu>
void qSort(size_t n, algorithmDataType * x, algorithmIndexType * index)
{
    // Avoid the threading layer call for 2-element arrays.
    if (n <= 2)
    {
        if (n == 2 && x[1] < x[0])
        {
            daal::services::internal::swap<cpu, algorithmDataType>(x[0], x[1]);
            daal::services::internal::swap<cpu, algorithmIndexType>(index[0], index[1]);
        }
        return;
    }
    daal::parallel_index_sort(n, x, index);
}

/**
 *  \brief Sorts array x and rearranges arrays w and z accordingly
 *
 *  \param n[in] Length of input arrays
 *  \param x     Array that is used as "key" when sorted
 *  \param w     Array that is used as "value" when sorted
 *  \param z     Array that is used as "value" when sorted
 */
template <typename algorithmFPtype, typename wType, typename zType, CpuType cpu>
void qSort(size_t n, algorithmFPtype * x, wType * w, zType * z)
{
    daal::parallel_index_sort(n, x, w, z);
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
