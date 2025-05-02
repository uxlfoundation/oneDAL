/* file: service_utils.h */
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
//  Declaration of service utilities
//--
*/
#ifndef __SERVICE_UTILS_H__
#define __SERVICE_UTILS_H__

#include "services/env_detect.h"
#include "src/services/service_type_traits.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace services
{
namespace internal
{
template <CpuType cpu, typename T>
inline typename RemoveReference<cpu, T>::type && move(T && object)
{
    return static_cast<typename RemoveReference<cpu, T>::type &&>(object);
}

template <CpuType cpu, typename T>
inline T && forward(typename RemoveReference<cpu, T>::type & object)
{
    return static_cast<T &&>(object);
}

template <typename T, CpuType cpu>
inline T * addressOf(T & value)
{
    return reinterpret_cast<T *>(&const_cast<char &>(reinterpret_cast<const volatile char &>(value)) // const_cast
    );                                                                                               // reinterpret_cast
}

template <CpuType cpu, typename T>
inline void swap(T & x, T & y)
{
    T tmp = x;
    x     = y;
    y     = tmp;
}

template <CpuType cpu, typename ForwardIterator1, typename ForwardIterator2>
DAAL_FORCEINLINE void iterSwap(ForwardIterator1 a, ForwardIterator2 b)
{
    swap<cpu>(*a, *b);
}

template <CpuType cpu, typename ForwardIterator, typename T>
ForwardIterator lowerBound(ForwardIterator first, ForwardIterator last, const T & value)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it              = first;
        const auto step = count / 2;
        it += step; // advance.
        if (*it < value)
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator, typename T, typename Compare>
ForwardIterator lowerBound(ForwardIterator first, ForwardIterator last, const T & value, Compare compare)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it              = first;
        const auto step = count / 2;
        it += step; // advance.
        if (compare(*it, value))
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator, typename T>
ForwardIterator upperBound(ForwardIterator first, ForwardIterator last, const T & value)
{
    ForwardIterator it;
    auto count = last - first; // distance.
    while (count > 0)
    {
        it              = first;
        const auto step = count / 2;
        it += step; // advance.
        if (!(value < *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator, typename T, typename Compare>
ForwardIterator upperBound(ForwardIterator first, ForwardIterator last, const T & value, Compare compare)
{
    auto count = last - first; // distance.
    if (count > 0)
    {
        if (compare(value, *first)) return first;
        if (count >= 2)
        {
            ForwardIterator second = first;
            ++second;
            if (compare(value, *second)) return second;
        }
    }
    ForwardIterator it;
    while (count > 0)
    {
        it              = first;
        const auto step = count / 2;
        it += step; // advance.
        if (!compare(value, *it))
        {
            first = ++it;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

/* Converts double input NaNs\Infs to output positive "big" value */
template <CpuType cpu>
inline double infToBigValue(double arg)
{
    uint64_t lBigValue = 0x7fefffffffffffff;

    if (((_daal_dp_union_t *)&arg)->bits.exponent == 0x7FF) // infinite number (inf or nan)
    {
        return *(double *)&lBigValue;
    }
    else
    {
        return arg;
    }
}

/* Converts float input NaNs\Infs to output positive "big" value */
template <CpuType cpu>
inline float infToBigValue(float arg)
{
    uint32_t iBigValue = 0x7e7fffff;

    if (((_daal_sp_union_t *)&arg)->bits.exponent == 0xFF) // infinite number (inf or nan)
    {
        return *(float *)&iBigValue;
    }
    else
    {
        return arg;
    }
}

template <CpuType cpu, typename T>
inline const T & min(const T & a, const T & b)
{
    return !(b < a) ? a : b;
}

template <CpuType cpu, typename T>
inline const T & max(const T & a, const T & b)
{
    return (a < b) ? b : a;
}

template <CpuType cpu, typename BidirectionalIterator, typename Compare>
BidirectionalIterator partition(BidirectionalIterator first, BidirectionalIterator last, Compare compare)
{
    while (first != last)
    {
        while (compare(*first))
        {
            if (++first == last) return first;
        }
        do
        {
            if (--last == first) return first;
        } while (!compare(*last));
        swap<cpu>(*first, *last);
        ++first;
    }
    return first;
}

template <CpuType cpu, typename ForwardIterator>
ForwardIterator maxElement(ForwardIterator first, ForwardIterator last)
{
    if (first == last)
    {
        return last;
    }
    auto largest = first;
    while (++first != last)
    {
        if (*largest < *first)
        {
            largest = first;
        }
    }
    return largest;
}

template <CpuType cpu, typename ForwardIterator, typename Compare>
ForwardIterator maxElement(ForwardIterator first, ForwardIterator last, Compare compare)
{
    if (first == last)
    {
        return last;
    }
    auto largest = first;
    while (++first != last)
    {
        if (compare(*largest, *first))
        {
            largest = first;
        }
    }
    return largest;
}

template <typename algorithmFPType, CpuType cpu>
void transpose(const algorithmFPType * src, size_t rows, size_t cols, algorithmFPType * dst)
{
    for (size_t j = 0; j < cols; j++)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < rows; i++)
        {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
size_t getMaxElementIndex(const algorithmFPType * val, size_t n)
{
    DAAL_ASSERT(n > 0);
    algorithmFPType maxVal = val[0];
    size_t maxIdx          = 0;
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 1; i < n; ++i)
    {
        if (maxVal < val[i])
        {
            maxVal = val[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Service function, memcpy src to dst
//////////////////////////////////////////////////////////////////////////////////////////
template <typename T, CpuType cpu>
inline void tmemcpy(T * dst, const T * src, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

} // namespace internal
} // namespace services
} // namespace daal

#endif
