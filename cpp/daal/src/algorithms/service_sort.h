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

typedef int (*CompareFunction)(const void *, const void *);

/**
 * \brief Quick sort function that sorts array x
 *
 * \param n[in]     Length of input arrays
 * \param x[in,out] Array to sort
 */
template <typename algorithmDataType, CpuType cpu>
void qSort(size_t n, algorithmDataType * x)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmDataType a;
    const int M = 7, NSTACK = 128;
    int istack[NSTACK];

    ir = n - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = x[j];

                for (i = j - 1; i >= l; i--)
                {
                    if (x[i] <= a)
                    {
                        break;
                    }
                    x[i + 1] = x[i];
                }

                x[i + 1] = a;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmDataType>(x[k], x[l + 1]);
            if (x[l] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[ir]);
            }
            if (x[l + 1] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l + 1], x[ir]);
            }
            if (x[l] > x[l + 1])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for (;;)
            {
                while (x[++i] < a)
                    ;
                while (x[--j] > a)
                    ;
                if (j < i)
                {
                    break;
                }
                daal::services::internal::swap<cpu, algorithmDataType>(x[i], x[j]);
            }
            x[l + 1] = x[j];

            x[j] = a;
            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}

/**
 * \brief Quick sort function that sorts array x
 *
 * \param n[in]       Length of input arrays
 * \param x[in,out]   Array to sort
 * \param compare[in] Pointer to compare function
 */
template <typename algorithmDataType, CpuType cpu>
void qSort(size_t n, algorithmDataType * x, CompareFunction compare)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmDataType a;
    const int M = 7, NSTACK = 128;
    int istack[NSTACK];

    ir = n - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = x[j];

                for (i = j - 1; i >= l; i--)
                {
                    if (compare(x + i, &a) < 1)
                    {
                        break;
                    }
                    x[i + 1] = x[i];
                }

                x[i + 1] = a;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmDataType>(x[k], x[l + 1]);

            if (compare(x + l, x + ir) == 1)
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[ir]);
            }
            if (compare(x + l + 1, x + ir) == 1)
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l + 1], x[ir]);
            }
            if (compare(x + l, x + l + 1) == 1)
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for (;;)
            {
                while (compare(&x[++i], &a) == -1)
                    ;
                while (compare(&x[--j], &a) == 1)
                    ;
                if (j < i)
                {
                    break;
                }
                daal::services::internal::swap<cpu, algorithmDataType>(x[i], x[j]);
            }
            x[l + 1] = x[j];

            x[j] = a;
            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}

template <typename T, CpuType cpu>
void qSortByKey(size_t n, T * x)
{
    int i, ir, j, k, jstack = -1, l = 0;
    T a;
    const int M = 7, NSTACK = 128;
    int istack[NSTACK];

    ir = n - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = x[j];

                for (i = j - 1; i >= l; i--)
                {
                    if (x[i].key <= a.key)
                    {
                        break;
                    }
                    x[i + 1] = x[i];
                }

                x[i + 1] = a;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, T>(x[k], x[l + 1]);

            if (x[l].key > x[ir].key)
            {
                daal::services::internal::swap<cpu, T>(x[l], x[ir]);
            }
            if (x[l + 1].key > x[ir].key)
            {
                daal::services::internal::swap<cpu, T>(x[l + 1], x[ir]);
            }
            if (x[l].key > x[l + 1].key)
            {
                daal::services::internal::swap<cpu, T>(x[l], x[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            for (;;)
            {
                while (x[++i].key < a.key)
                    ;
                while (x[--j].key > a.key)
                    ;

                if (j < i)
                {
                    break;
                }
                daal::services::internal::swap<cpu, T>(x[i], x[j]);
            }
            x[l + 1] = x[j];

            x[j] = a;
            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
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
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmDataType a;
    algorithmIndexType b;
    const int M = 7, NSTACK = 128;
    algorithmDataType istack[NSTACK];

    ir = n - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = x[j];
                b = index[j];

                for (i = j - 1; i >= l; i--)
                {
                    if (x[i] <= a)
                    {
                        break;
                    }
                    x[i + 1]     = x[i];
                    index[i + 1] = index[i];
                }

                x[i + 1]     = a;
                index[i + 1] = b;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmDataType>(x[k], x[l + 1]);
            daal::services::internal::swap<cpu, algorithmIndexType>(index[k], index[l + 1]);
            if (x[l] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[ir]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[l], index[ir]);
            }
            if (x[l + 1] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l + 1], x[ir]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[l + 1], index[ir]);
            }
            if (x[l] > x[l + 1])
            {
                daal::services::internal::swap<cpu, algorithmDataType>(x[l], x[l + 1]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[l], index[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            b = index[l + 1];
            for (;;)
            {
                while (x[++i] < a)
                    ;
                while (x[--j] > a)
                    ;
                if (j < i)
                {
                    break;
                }
                daal::services::internal::swap<cpu, algorithmDataType>(x[i], x[j]);
                daal::services::internal::swap<cpu, algorithmIndexType>(index[i], index[j]);
            }
            x[l + 1]     = x[j];
            index[l + 1] = index[j];

            x[j]     = a;
            index[j] = b;
            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}

template <typename algorithmFPtype, typename wType, typename zType, CpuType cpu>
void qSort(size_t n, algorithmFPtype * x, wType * w, zType * z)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmFPtype a;
    wType b;
    zType c;
    const int M = 7, NSTACK = 128;
    algorithmFPtype istack[NSTACK];

    ir = n - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = x[j];
                b = w[j];
                c = z[j];

                for (i = j - 1; i >= l; i--)
                {
                    if (x[i] <= a)
                    {
                        break;
                    }
                    x[i + 1] = x[i];
                    w[i + 1] = w[i];
                    z[i + 1] = z[i];
                }

                x[i + 1] = a;
                w[i + 1] = b;
                z[i + 1] = c;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmFPtype>(x[k], x[l + 1]);
            daal::services::internal::swap<cpu, wType>(w[k], w[l + 1]);
            daal::services::internal::swap<cpu, zType>(z[k], z[l + 1]);
            if (x[l] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l], x[ir]);
                daal::services::internal::swap<cpu, wType>(w[l], w[ir]);
                daal::services::internal::swap<cpu, zType>(z[l], z[ir]);
            }
            if (x[l + 1] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l + 1], x[ir]);
                daal::services::internal::swap<cpu, wType>(w[l + 1], w[ir]);
                daal::services::internal::swap<cpu, zType>(z[l + 1], z[ir]);
            }
            if (x[l] > x[l + 1])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l], x[l + 1]);
                daal::services::internal::swap<cpu, wType>(w[l], w[l + 1]);
                daal::services::internal::swap<cpu, zType>(z[l], z[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            b = w[l + 1];
            c = z[l + 1];
            for (;;)
            {
                while (x[++i] < a)
                    ;
                while (x[--j] > a)
                    ;
                if (j < i)
                {
                    break;
                }
                daal::services::internal::swap<cpu, algorithmFPtype>(x[i], x[j]);
                daal::services::internal::swap<cpu, wType>(w[i], w[j]);
                daal::services::internal::swap<cpu, zType>(z[i], z[j]);
            }
            x[l + 1] = x[j];
            w[l + 1] = w[j];
            z[l + 1] = z[j];

            x[j] = a;
            w[j] = b;
            z[j] = c;
            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}

template <typename algorithmFPtype, CpuType cpu>
void indexBubbleSortDesc(services::Collection<algorithmFPtype> & x, services::Collection<algorithmFPtype> & idx1,
                         services::Collection<algorithmFPtype> & idx2)
{
    size_t n = x.size();
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n - i - 1; j++)
        {
            if (x[j] < x[j + 1])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[j], x[j + 1]);
                daal::services::internal::swap<cpu, algorithmFPtype>(idx1[j], idx1[j + 1]);
                daal::services::internal::swap<cpu, algorithmFPtype>(idx2[j], idx2[j + 1]);
            }
        }
    }
}

#if defined(DAAL_INTEL_CPP_COMPILER)
    #define __RADIX_SORT_CAST32(x) (_castf32_u32(x))
    #define __RADIX_SORT_CAST64(x) (_castf64_u64(x))
#else
    #define __RADIX_SORT_CAST32(x) (*reinterpret_cast<const unsigned int *>(&(x)))
    #define __RADIX_SORT_CAST64(x) (*reinterpret_cast<const DAAL_UINT64 *>(&(x)))
#endif

#define DAAL_INSERTION_SORT_MAX_SIZE_IN_INTROSORT 32
#define DAAL_MIDDLE_OF_3_THRESHOLD                41

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
static DAAL_FORCEINLINE void medianOf3(RandomAccessIterator first, RandomAccessIterator mid, RandomAccessIterator last, Compare compare)
{
    if (compare(*mid, *first))
    {
        iterSwap<cpu>(mid, first);
    }
    if (compare(*last, *mid))
    {
        iterSwap<cpu>(last, mid);

        // Middle changed - need to to compare it with first again.
        if (compare(*mid, *first))
        {
            iterSwap<cpu>(mid, first);
        }
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
DAAL_FORCEINLINE void partition3(RandomAccessIterator first, RandomAccessIterator last, RandomAccessIterator & partFirst,
                                 RandomAccessIterator & partLast, Compare compare)
{
    RandomAccessIterator mid = first + (last - first) / 2;
    if (DAAL_MIDDLE_OF_3_THRESHOLD < last - first)
    {
        const auto step = (last - first) / 8;
        medianOf3<cpu>(first, first + step, first + 2 * step, compare);
        medianOf3<cpu>(mid - step, mid, mid + step, compare);
        medianOf3<cpu>(last - 2 * step - 1, last - step - 1, last - 1, compare);
        medianOf3<cpu>(first + step, mid, last - step - 1, compare);
    }
    else
    {
        medianOf3<cpu>(first, mid, last - 1, compare);
    }

    partFirst = mid;
    partLast  = partFirst + 1;

    while (first < partFirst && !compare(*(partFirst - 1), *partFirst) && !compare(*partFirst, *(partFirst - 1)))
    {
        --partFirst;
    }
    while (partLast < last && !compare(*partLast, *partFirst) && !compare(*partFirst, *partLast))
    {
        ++partLast;
    }

    RandomAccessIterator f = partLast;
    RandomAccessIterator l = partFirst;

    for (;;)
    {
        for (; f < last; ++f)
        {
            if (!compare(*partFirst, *f))
            {
                if (compare(*f, *partFirst))
                {
                    break;
                }
                if (partLast != f)
                {
                    iterSwap<cpu>(partLast, f);
                }
                ++partLast;
            }
        }

        for (; first < l; --l)
        {
            if (!compare(*(l - 1), *partFirst))
            {
                if (compare(*partFirst, *(l - 1)))
                {
                    break;
                }
                if (--partFirst != l - 1)
                {
                    iterSwap<cpu>(partFirst, l - 1);
                }
            }
        }

        if (l == first && f == last)
        {
            break;
        }

        if (l == first)
        {
            if (partLast != f)
            {
                iterSwap<cpu>(partFirst, partLast);
            }
            ++partLast;
            iterSwap<cpu>(partFirst, f);
            ++partFirst;
            ++f;
        }
        else if (f == last)
        {
            if (--l != --partFirst)
            {
                iterSwap<cpu>(l, partFirst);
            }
            iterSwap<cpu>(partFirst, --partLast);
        }
        else
        {
            iterSwap<cpu>(f, --l);
            ++f;
        }
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Diff, typename Compare>
static void internalIntroSort(RandomAccessIterator first, RandomAccessIterator last, Diff depthLimit, Compare compare)
{
    auto count = last - first;
    while (DAAL_INSERTION_SORT_MAX_SIZE_IN_INTROSORT < count && 0 < depthLimit)
    {
        RandomAccessIterator partFirst, partLast;
        partition3<cpu>(first, last, partFirst, partLast, compare);

        depthLimit /= 2;
        depthLimit += depthLimit / 2;
        if (partFirst - first < partLast - last)
        {
            internalIntroSort<cpu>(first, partFirst, depthLimit, compare);
            first = partLast;
        }
        else
        {
            internalIntroSort<cpu>(partLast, last, depthLimit, compare);
            last = partFirst;
        }
        count = last - first;
    }

    if (DAAL_INSERTION_SORT_MAX_SIZE_IN_INTROSORT < count)
    {
        makeMaxHeap<cpu>(first, last, compare);
        sortMaxHeap<cpu>(first, last, compare);
    }
    else if (1 < count)
    { // Insertion sort.
        for (RandomAccessIterator next = first; ++next != last;)
        {
            auto value = *next; // Moving can be used instead.
            if (compare(value, *first))
            {
                // Move backward
                for (RandomAccessIterator result = next, i = next; first != i; --result)
                {
                    *result = *(--i); // Moving can be used instead.
                }

                *first = value; // Moving can be used instead.
            }
            else
            {
                RandomAccessIterator result = next;
                for (RandomAccessIterator i = next; compare(value, *(--i)); result = i)
                {
                    *result = *i; // Moving can be used instead.
                }
                *result = value; // Moving can be used instead.
            }
        }
    }
}

template <CpuType cpu, typename RandomAccessIterator, typename Compare>
DAAL_FORCEINLINE void introSort(RandomAccessIterator first, RandomAccessIterator last, Compare compare)
{
    internalIntroSort<cpu>(first, last, last - first, compare);
}

template <CpuType cpu, typename ForwardIterator, typename Compare>
ForwardIterator isSortedUntil(ForwardIterator first, ForwardIterator last, Compare compare)
{
    if (first != last)
    {
        for (auto next = first; ++next != last; ++first)
        {
            if (compare(*next, *first))
            {
                return next;
            }
        }
    }
    return last;
}

template <CpuType cpu, typename ForwardIterator, typename Compare>
DAAL_FORCEINLINE bool isSorted(ForwardIterator first, ForwardIterator last, Compare compare)
{
    return (isSortedUntil<cpu>(first, last, compare) == last);
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
