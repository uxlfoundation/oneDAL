/* file: buffer.h */
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
// Declaration and implementation of the shared pointer class.
//--
*/

#ifndef __DAAL_SERVICES_BUFFER_H__
#define __DAAL_SERVICES_BUFFER_H__

#if (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0

    #include "services/internal/buffer_impl.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace interface1
{
/**
 * @ingroup sycl
 * @{
 */

/**
 *  <a name="DAAL-CLASS-SERVICES__BUFFER"></a>
 *  \brief Wrapper for a SYCL* buffer
 *  Can hold data on the host side using shared pointer,
 *  or on host/device sides using SYCL* buffer
 *  \DAAL_DEPRECATED
 */
template <typename T>
class Buffer : public Base
{
public:
    /**
     *   Creates empty Buffer object
     */
    Buffer() {}

    /**
     *   Creates a Buffer object from host-allocated raw pointer
     *   Buffer does not own this pointer
     */
    Buffer(T * data, size_t size, Status & status) : _impl(internal::HostBuffer<T>::create(data, size, status)) {}

    #ifndef DAAL_NOTHROW_EXCEPTIONS
    /**
     *   Creates a Buffer object from host-allocated raw pointer
     *   Buffer does not own this pointer
     */
    Buffer(T * data, size_t size)
    {
        Status status;
        _impl.reset(internal::HostBuffer<T>::create(data, size, status));
        throwIfPossible(status);
    }
    #endif // DAAL_NOTHROW_EXCEPTIONS

    /**
     *   Creates a Buffer object referencing the shared pointer to the host-allocated data
     */
    Buffer(const SharedPtr<T> & data, size_t size, Status & status) : _impl(internal::HostBuffer<T>::create(data, size, status)) {}

    #ifndef DAAL_NOTHROW_EXCEPTIONS
    /**
     *   Creates a Buffer object referencing the shared pointer to the host-allocated data
     */
    Buffer(const SharedPtr<T> & data, size_t size)
    {
        Status status;
        _impl.reset(internal::HostBuffer<T>::create(data, size, status));
        throwIfPossible(status);
    }
    #endif // DAAL_NOTHROW_EXCEPTIONS

    /**
     *  Returns true if Buffer points to any data
     */
    operator bool() const { return _impl; }

    /**
     *  Returns true if Buffer is equal to \p other
     */
    bool operator==(const Buffer & other) const { return _impl.get() == other._impl.get(); }

    /**
     *  Returns true if Buffer is not equal to \p other
     */
    bool operator!=(const Buffer & other) const { return _impl.get() != other._impl.get(); }

    /**
     *  Converts data inside the buffer to the host side
     *  \param[in]  rwFlag  Access flag to the data
     *  \param[out] status  Status of operation
     *  \return host-allocated shared pointer to the data
     */
    SharedPtr<T> toHost(const data_management::ReadWriteMode & rwFlag, Status & status) const
    {
        if (!_impl)
        {
            status |= ErrorEmptyBuffer;
            return SharedPtr<T>();
        }
        return internal::HostBufferConverter<T>().toHost(*_impl, rwFlag, status);
    }

    #ifndef DAAL_NOTHROW_EXCEPTIONS
    /**
     *  Converts data inside the buffer to the host side, throws exception if conversion fails
     *  \param[in]  rwFlag  Access flag to the data
     *  \return host-allocated shared pointer to the data
     */
    SharedPtr<T> toHost(const data_management::ReadWriteMode & rwFlag) const
    {
        Status status;
        const SharedPtr<T> ptr = toHost(rwFlag, status);
        throwIfPossible(status);
        return ptr;
    }
    #endif // DAAL_NOTHROW_EXCEPTIONS

    /**
     *   Returns the total number of elements in the buffer
     */
    size_t size() const
    {
        if (!_impl)
        {
            return 0;
        }
        return _impl->size();
    }

    /**
     *   Drops underlying reference to the data from the buffer and makes it empty
     */
    void reset() { _impl.reset(); }

    /**
     *  Creates Buffer object that points to the same memory as a parent but with offset
     *  \param[in]  offset  Offset in elements from start of the parent buffer
     *  \param[in]  size    Number of elements in the sub-buffer
     *  \param[out] status  Status of operation
     *  \return Buffer that contains only a part of the original buffer
     */
    Buffer<T> getSubBuffer(size_t offset, size_t size, Status & status) const
    {
        if (!_impl)
        {
            status |= ErrorEmptyBuffer;
            return Buffer<T>();
        }
        return Buffer<T>(_impl->getSubBuffer(offset, size, status));
    }

    #ifndef DAAL_NOTHROW_EXCEPTIONS
    /**
     *  Creates Buffer object that points to the same memory as a parent but with offset,
     *  throws exception if conversion fails
     *  \param[in]  offset  Offset in elements from start of the parent buffer
     *  \param[in]  size    Number of elements in the sub-buffer
     *  \return Buffer that contains only a part of the original buffer
     */
    Buffer<T> getSubBuffer(size_t offset, size_t size) const
    {
        Status status;
        const Buffer<T> suBuffer = getSubBuffer(offset, size, status);
        throwIfPossible(status);
        return suBuffer;
    }
    #endif // DAAL_NOTHROW_EXCEPTIONS

private:
    explicit Buffer(internal::BufferIface<T> * impl) : _impl(impl) {}
    explicit Buffer(const SharedPtr<internal::BufferIface<T> > & impl) : _impl(impl) {}

    SharedPtr<internal::BufferIface<T> > _impl;
};

/** @} */
} // namespace interface1

using interface1::Buffer;

} // namespace internal
} // namespace services
} // namespace daal

#endif // (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0

#endif
