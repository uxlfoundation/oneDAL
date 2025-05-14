/* file: service_unique_ptr.h */
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

#ifndef __SERVICE_UNIQUE_PTR_H__
#define __SERVICE_UNIQUE_PTR_H__

#include "src/services/service_utils.h"
#include "src/services/service_deleters.h"

namespace daal
{
namespace internal
{
/* STL compatible unique_ptr implementation (doesn't handle the case when T is an array) */
template <typename T, CpuType cpu, typename Deleter = services::internal::DefaultDeleter<T, cpu> >
class UniquePtr
{
private:
    T * _object;
    Deleter _deleter;

public:
    UniquePtr() : _object(nullptr) {}

    explicit UniquePtr(T * object) : _object(object) {}

    template <typename UDeleter>
    UniquePtr(T * object, const UDeleter & deleter) : _object(object), _deleter(deleter)
    {}

    template <typename U, typename UDeleter>
    UniquePtr(UniquePtr<U, cpu, UDeleter> && other)
        : _object(other.release()), _deleter(services::internal::forward<cpu, UDeleter>(other.getDeleter()))
    {}

    ~UniquePtr() { reset(); }

    inline T * get() const { return _object; }
    inline operator bool() const { return _object != nullptr; }

    inline T & operator*() const { return *_object; }
    inline T * operator->() const { return _object; }

    template <typename U, typename UDeleter>
    UniquePtr & operator=(UniquePtr<U, cpu, UDeleter> && other)
    {
        reset(other.release());
        _deleter = services::internal::move<cpu, UDeleter>(other.getDeleter());
        return *this;
    }

    inline void reset(T * object = nullptr)
    {
        if (_object)
        {
            _deleter(_object);
        }
        _object = object;
    }

    inline T * release()
    {
        T * result = _object;
        _object    = nullptr;
        return result;
    }

    inline Deleter & getDeleter() { return _deleter; }
    inline const Deleter & getDeleter() const { return _deleter; }

    template <typename U, typename UDeleter>
    UniquePtr(const UniquePtr<U, cpu, UDeleter> &) = delete;

    template <typename U, typename UDeleter>
    UniquePtr & operator=(const UniquePtr<U, cpu, UDeleter> &) = delete;
};

// Creates UniquePtr<T, cpu> by calling constructor of T with the given arguments
// Usage: auto object = makeUnique<T, cpu>(arg_1, ..., arg_2);
template <typename T, CpuType cpu, typename... Args>
UniquePtr<T, cpu> makeUnique(Args &&... args)
{
    using namespace daal::services::internal;
    return UniquePtr<T, cpu>(new T(services::internal::forward<cpu, Args>(args)...));
}

} // namespace internal
} // namespace daal

#endif
