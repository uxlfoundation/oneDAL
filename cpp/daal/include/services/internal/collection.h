/* file: collection.h */
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

#ifndef __SERVICES_INTERNAL_COLLECTION_H__
#define __SERVICES_INTERNAL_COLLECTION_H__

#include "services/base.h"
#include "services/buffer_view.h"
#include "services/collection.h"
#include "services/internal/error_handling_helpers.h"

namespace daal
{
namespace services
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-SERVICES__INTERNAL__PRIMITIVECOLLECTION"></a>
 * \brief  Class that provides simple memory management routines for handling blocks
 *         of continues memory, also provides automatic memory deallocation. Note this
 *         class doesn't provide functionality for objects constructions and simply allocates
 *         and deallocates memory. In case of objects consider Collection or ObjectPtrCollection
 * \tparam T Type of elements which are stored in the buffer
 */
template <typename T>
class PrimitiveCollection : public Base
{
public:
    PrimitiveCollection() : _buffer(NULL), _size(0) {}

    explicit PrimitiveCollection(size_t size, services::Status * status = NULL) : _buffer(NULL), _size(0)
    {
        services::Status localStatus = reallocate(size);
        services::internal::tryAssignStatusAndThrow(status, localStatus);
    }

    virtual ~PrimitiveCollection() { destroy(); }

    void destroy()
    {
        services::daal_free((void *)_buffer);
        _buffer = NULL;
        _size   = 0;
    }

    services::Status reallocate(size_t size, bool copy = false)
    {
        if (_size == size)
        {
            return services::Status();
        }

        T * buffer = (T *)services::daal_malloc(sizeof(T) * size);
        if (!buffer)
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        if (copy)
        {
            // Copy data from old buffer to new buffer, preserving existing data
            // Use min of old and new size to prevent buffer overrun
            const size_t copy_count = (size < _size) ? size : _size;
            for (size_t i = 0; i < copy_count; i++)
            {
                buffer[i] = _buffer[i];
            }
        }

        destroy();

        _size   = size;
        _buffer = buffer;
        return services::Status();
    }

    services::Status enlarge(size_t factor = 2, bool copy = false) { return reallocate(_size * factor, copy); }

    size_t size() const { return _size; }

    T * data() const { return _buffer; }

    T * offset(size_t elementsOffset) const
    {
        DAAL_ASSERT(elementsOffset <= _size);
        return _buffer + elementsOffset;
    }

    T & operator[](size_t index)
    {
        DAAL_ASSERT(index < _size);
        return _buffer[index];
    }

    const T & operator[](size_t index) const
    {
        DAAL_ASSERT(index < _size);
        return _buffer[index];
    }

    services::BufferView<T> view() const { return services::BufferView<T>(_buffer, _size); }

private:
    PrimitiveCollection(const PrimitiveCollection &);
    PrimitiveCollection & operator=(const PrimitiveCollection &);

private:
    T * _buffer;
    size_t _size;
};

/**
 * <a name="DAAL-CLASS-SERVICES__INTERNAL__OBJECTPTRCOLLECTION"></a>
 * \brief  Class that implements functionality of collection container and holds pointers
 *         to objects of specified type, also provides automatic objects disposal
 * \tparam T Type of objects which are stored in the container
 * \tparam Deleter Type of deleter to be called on collection disposal
 */
template <typename T, typename Deleter = ObjectDeleter<T> >
class ObjectPtrCollection : public Base
{
public:
    ObjectPtrCollection() {}

    ObjectPtrCollection(const Deleter & deleter) : _deleter(deleter) {}

    virtual ~ObjectPtrCollection()
    {
        for (size_t i = 0; i < _objects.size(); i++)
        {
            _deleter((const void *)_objects[i]);
        }
    }

    T & operator[](size_t index) const
    {
        DAAL_ASSERT(index < _objects.size());
        return *(_objects[index]);
    }

    size_t size() const { return _objects.size(); }

    bool push_back(T * object)
    {
        if (!object)
        {
            return false;
        }

        return _objects.safe_push_back(object);
    }

    template <typename U>
    bool safe_push_back()
    {
        return _objects.push_back(new U());
    }

private:
    ObjectPtrCollection(const ObjectPtrCollection &);
    ObjectPtrCollection & operator=(const ObjectPtrCollection &);

private:
    Deleter _deleter;
    services::Collection<T *> _objects;
};

/**
 *  <a name="DAAL-CLASS-SERVICES__INTERNAL__HEAPALLOCATABLECOLLECTION"></a>
 *  \brief   Wrapper for services::Collection that allocates and deallocates
 *           memory using internal new/delete operators
 *  \tparam  T  Type of an object stored in the container
 */
template <typename T>
class HeapAllocatableCollection : public Base, public services::Collection<T>
{
public:
    static SharedPtr<HeapAllocatableCollection<T> > create(services::Status * status = NULL)
    {
        typedef SharedPtr<HeapAllocatableCollection<T> > PtrType;

        HeapAllocatableCollection<T> * collection = new internal::HeapAllocatableCollection<T>();
        if (!collection)
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorMemoryAllocationFailed);
            return PtrType();
        }

        return PtrType(collection);
    }

    static SharedPtr<HeapAllocatableCollection<T> > create(size_t n, services::Status * status = NULL)
    {
        typedef SharedPtr<HeapAllocatableCollection<T> > PtrType;

        HeapAllocatableCollection<T> * collection = new internal::HeapAllocatableCollection<T>(n);
        if (!collection || !collection->data())
        {
            delete collection;
            services::internal::tryAssignStatusAndThrow(status, services::ErrorMemoryAllocationFailed);
            return PtrType();
        }

        return PtrType(collection);
    }

    HeapAllocatableCollection() {}

    explicit HeapAllocatableCollection(size_t n) : services::Collection<T>(n) {}
};

/**
 *  <a name="DAAL-CLASS-SERVICES__INTERNAL__COLLECTIONPTR"></a>
 *  \brief   Shared pointer to the Collection object
 *  \tparam  T  Type of an object stored in the container
 */
template <class T>
class CollectionPtr : public SharedPtr<HeapAllocatableCollection<T> >
{
private:
    typedef SharedPtr<HeapAllocatableCollection<T> > super;

public:
    CollectionPtr() {}

    template <class U>
    CollectionPtr(const SharedPtr<U> & other) : super(other)
    {}

    template <class U>
    explicit CollectionPtr(U * ptr) : super(ptr)
    {}

    template <class U, class D>
    explicit CollectionPtr(U * ptr, const D & deleter) : super(ptr, deleter)
    {}
};

} // namespace internal
} // namespace services
} // namespace daal

#endif
