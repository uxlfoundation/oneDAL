/* file: service_array.h */
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
//  Service classes for CPU-specified arrays
//--
*/

#ifndef __SERVICE_ARRAY__
#define __SERVICE_ARRAY__

#include "src/externals/service_memory.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{
template <CpuType cpu>
class DefaultAllocator
{
public:
    static void * alloc(size_t nBytes) { return services::daal_calloc(nBytes); }
    static void free(void * ptr) { services::daal_free(ptr); }
};

template <CpuType cpu>
class ScalableAllocator
{
public:
    static void * alloc(size_t nBytes) { return services::internal::service_scalable_calloc<byte, cpu>(nBytes); }
    static void free(void * ptr) { services::internal::service_scalable_free<byte, cpu>((byte *)ptr); }
};

//Simple container
template <typename T, CpuType cpu, typename Allocator = DefaultAllocator<cpu> >
class TVector
{
public:
    DAAL_NEW_DELETE();
    TVector(size_t n = 0) : _data(nullptr), _size(0)
    {
        if (n) alloc(n);
    }
    TVector(size_t n, T val) : _data(nullptr), _size(0)
    {
        if (n)
        {
            alloc(n);
            for (size_t i = 0; i < n; ++i) _data[i] = val;
        }
    }
    ~TVector() { destroy(); }
    TVector(const TVector & o) : _data(nullptr), _size(0)
    {
        if (o._size)
        {
            alloc(o._size);
            for (size_t i = 0; i < _size; ++i) _data[i] = o._data[i];
        }
    }

    TVector & operator=(const TVector & o)
    {
        if (this != &o)
        {
            if (_size < o._size)
            {
                destroy();
                alloc(o._size);
            }
            for (size_t i = 0; i < _size; ++i) _data[i] = o._data[i];
        }
        return *this;
    }

    size_t size() const { return _size; }

    void setValues(size_t n, T val)
    {
        DAAL_ASSERT(n <= size());
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < n; ++i) _data[i] = val;
    }

    void setAll(T val)
    {
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _size; ++i) _data[i] = val;
    }

    void pushBack(T val)
    {
        resize(_size + 1);
        _data[_size - 1] = val;
    }

    void resize(size_t n)
    {
        T * ptr = (T *)Allocator::alloc(n * sizeof(T));
        services::internal::daal_memcpy_s(ptr, n * sizeof(T), _data, _size * sizeof(T));
        Allocator::free(_data);
        _data = ptr;
        _size = n;
    }

    void reset(size_t n)
    {
        if (n != _size)
        {
            destroy();
            alloc(n);
        }
    }

    void resize(size_t n, T val)
    {
        reset(n);
        setAll(val);
    }

    T & operator[](size_t index)
    {
        DAAL_ASSERT(index < size());
        return _data[index];
    }

    const T & operator[](size_t index) const
    {
        DAAL_ASSERT(index < size());
        return _data[index];
    }
    T * detach()
    {
        auto res = _data;
        _data    = nullptr;
        _size    = 0;
        return res;
    }
    T * get() { return _data; }
    const T * get() const { return _data; }

private:
    void alloc(size_t n)
    {
        _data = (T *)(n ? Allocator::alloc(n * sizeof(T)) : nullptr);
        if (_data) _size = n;
    }

    void destroy()
    {
        if (_data)
        {
            Allocator::free(_data);
            _data = nullptr;
            _size = 0;
        }
    }

private:
    T * _data;
    size_t _size;
};

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
