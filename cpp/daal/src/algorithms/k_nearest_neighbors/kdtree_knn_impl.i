/* file: kdtree_knn_impl.i */
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
//  Common functions for K-Nearest Neighbors
//--
*/

#ifndef __KDTREE_KNN_IMPL_I__
#define __KDTREE_KNN_IMPL_I__

#if defined(_MSC_VER) || defined(DAAL_INTEL_CPP_COMPILER)
    #include <immintrin.h>
#endif

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace internal
{
using namespace daal::internal;

#define __KDTREE_MAX_NODE_COUNT_MULTIPLICATION_FACTOR 3
#define __KDTREE_LEAF_BUCKET_SIZE                     31 // Must be ((power of 2) minus 1).
#define __KDTREE_FIRST_PART_LEAF_NODES_PER_THREAD     3
#define __KDTREE_DIMENSION_SELECTION_SIZE             128
#define __KDTREE_MEDIAN_RANDOM_SAMPLE_COUNT           1024
#define __KDTREE_DEPTH_MULTIPLICATION_FACTOR          4
#define __KDTREE_SEARCH_SKIP                          32
#define __KDTREE_INDEX_VALUE_PAIRS_PER_THREAD         8192
#define __KDTREE_SAMPLES_PERCENT                      0.5
#define __KDTREE_MAX_SAMPLES                          1024
#define __KDTREE_MIN_SAMPLES                          256
#define __SIMDWIDTH                                   8

#define __KDTREE_NULLDIMENSION (static_cast<size_t>(-1))

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

template <typename T, CpuType cpu>
class Stack
{
public:
    Stack() : _data(nullptr) {}

    ~Stack()
    {
        services::daal_free(_data);
        _data = nullptr;
    }

    bool init(size_t size)
    {
        _data       = static_cast<T *>(services::internal::service_malloc<T, cpu>(size));
        _size       = size;
        _sizeMinus1 = size - 1;
        _top        = -1;
        _count      = 0;
        return _data;
    }

    void clear()
    {
        if (_data)
        {
            services::daal_free(_data);
            _data = nullptr;
        }
    }

    void reset()
    {
        _top   = -1;
        _count = 0;
    }

    DAAL_FORCEINLINE services::Status push(const T & value)
    {
        services::Status status;

        if (_count >= _size)
        {
            status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }

        _data[++_top] = value;
        ++_count;

        return status;
    }

    DAAL_FORCEINLINE T pop()
    {
        const T value = _data[_top--];
        --_count;
        return value;
    }

    bool empty() const { return (_count == 0); }

    size_t size() const { return _count; }

    services::Status grow()
    {
        _size *= 2;
        T * const newData = static_cast<T *>(services::internal::service_malloc<T, cpu>(_size));
        DAAL_CHECK_MALLOC(newData)
        _sizeMinus1 = _size - 1;
        int result  = services::internal::daal_memcpy_s(newData, _size * sizeof(T), _data, _count * sizeof(T));
        T * oldData = _data;
        _data       = newData;
        services::daal_free(oldData);
        oldData = nullptr;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

private:
    T * _data;
    size_t _top;
    size_t _count;
    size_t _size;
    size_t _sizeMinus1;
};

} // namespace internal
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
