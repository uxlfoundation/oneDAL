/* file: daal_shared_ptr.h */
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

#ifndef __DAAL_SHARED_PTR_H__
#define __DAAL_SHARED_PTR_H__

#include "services/base.h"
#include "services/daal_memory.h"
#include "services/error_id.h"
#include "services/daal_atomic_int.h"

namespace daal
{
namespace services
{
namespace interface1
{
/**
 * @defgroup memory Managing Memory
 * \brief Contains classes that implement memory allocation and deallocation.
 * @ingroup services
 * @{
 */
/**
 * <a name="DAAL-CLASS-SERVICES__DELETERIFACE"></a>
 * \brief Interface for a utility class used within SharedPtr to delete an object when the object owner is destroyed
 */
class DeleterIface
{
public:
    /**
     * Default destructor
     */
    virtual ~DeleterIface() {}

    /**
     * Deletes an object referenced by a pointer
     * \param[in]   ptr   Pointer to the object
     */
    virtual void operator()(const void * ptr) = 0;
};

/**
 * <a name="DAAL-CLASS-SERVICES__OBJECTDELETER"></a>
 * \brief Implementation of DeleterIface to destroy a pointer by the delete operator
 *
 * \tparam T    Class of the object to delete
 */
template <class T>
class ObjectDeleter : public DeleterIface
{
public:
    void operator()(const void * ptr) DAAL_C11_OVERRIDE { delete (const T *)(ptr); }
};

/**
 * <a name="DAAL-CLASS-SERVICES__SERVICEDELETER"></a>
 * \brief Implementation of DeleterIface to destroy a pointer by the daal_free function
 *
 * \tparam T    Class of the object to delete
 */
class ServiceDeleter : public DeleterIface
{
public:
    void operator()(const void * ptr) DAAL_C11_OVERRIDE { daal::services::daal_free((void *)ptr); }
};

/**
 * <a name="DAAL-CLASS-SERVICES__EMPTYDELETER"></a>
 * \brief Implementation of DeleterIface without pointer destroying
 *
 * \tparam T    Class of the object to delete
 */
class EmptyDeleter : public DeleterIface
{
public:
    void operator()(const void * /*ptr*/) DAAL_C11_OVERRIDE {}
};

/**
 * <a name="DAAL-CLASS-SERVICES__REFCOUNTER"></a>
 * \brief Implementation of reference counter
 *
 */
class DAAL_EXPORT RefCounter : public AtomicInt
{
public:
    /**
     * Default constructor
     */
    RefCounter() : AtomicInt(1) {}
    /** Destructor */
    virtual ~RefCounter() {}
    virtual void operator()(const void * ptr) = 0;
};

/**
 * <a name="DAAL-CLASS-SERVICES__REFCOUNTERIMP"></a>
 * \brief Provides implementations of the operator() method of the RefCounter class
 *
* \tparam Deleter    Class of the object to delete
 */
template <class Deleter>
class RefCounterImp : public RefCounter
{
public:
    /**
     * Default constructor
     */
    RefCounterImp() {}
    RefCounterImp(const Deleter & d) : _deleter(d) {}
    /** Destructor */
    virtual ~RefCounterImp() {}
    void operator()(const void * ptr) DAAL_C11_OVERRIDE { _deleter(ptr); }

protected:
    Deleter _deleter;
};

/**
 * <a name="DAAL-CLASS-SERVICES__SHAREDPTR"></a>
 * \brief Shared pointer that retains shared ownership of an object through a pointer.
 * Several SharedPtr objects may own the same object. The object is destroyed and its memory deallocated
 * when either of the following happens:\n
 * 1) the last remaining SharedPtr owning the object is destroyed.\n
 * 2) the last remaining SharedPtr owning the object is assigned another pointer via operator=.\n
 * The object is destroyed using the delete operator.
 *
 * \tparam T    Class of the managed object
 */
template <class T>
class SharedPtr
{
public:
    DAAL_NEW_DELETE();

    typedef T ElementType;

    /**
     * Constructs an empty shared pointer
     */
    SharedPtr() : _ownedPtr(NULL), _ptr(NULL), _refCount(NULL) {}

    /**
     * Constructs a shared pointer that manages an input pointer
     * \param[in] ptr   Pointer to manage
     */
    template <class U>
    explicit SharedPtr(U * ptr) : _ownedPtr((T *)ptr), _ptr((T *)ptr), _refCount(NULL)
    {
        if (_ownedPtr) _refCount = new RefCounterImp<ObjectDeleter<U> >();
    }

    /**
     * Constructs a shared pointer that manages an input pointer
     * \tparam U    Class of the managed object
     * \tparam D    Class of the deleter object
     * \param[in] ptr       Pointer to the managed object
     * \param[in] deleter   Object used to delete the pointer when the reference count becomes equal to zero
     */
    template <class U, class D>
    explicit SharedPtr(U * ptr, const D & deleter) : _ownedPtr((T *)ptr), _ptr((T *)ptr), _refCount(NULL)
    {
        if (_ownedPtr) _refCount = new RefCounterImp<D>(deleter);
    }

    SharedPtr(const SharedPtr<T> & ptr);

    SharedPtr(const SharedPtr<T> & ptr, T * shiftedPtr);

    /**
    * Constructs a shared pointer from another shared pointer of the different type
    * \param[in] other   Input shared pointer
    */
    template <class U>
    SharedPtr(const SharedPtr<U> & other);

    /**
     * Aliasing constructor: constructs a SharedPtr that shares ownership information with r,
     * but holds an unrelated and unmanaged ptr pointer.
     * Even if this SharedPtr is the last of the group to go out of scope,
     * it calls the destructor for the object originally managed by r.
     * However, calling get() on this always returns a copy of ptr.
     * It is the responsibility of a programmer to make sure this ptr remains valid
     * as long as this SharedPtr exists, such as in the typical use cases where ptr is a member of the object
     * managed by r or is an alias (e.g., downcast) of r.get()
     */
    template <class U>
    SharedPtr(const SharedPtr<U> & r, T * ptr, T * shiftedPtr);

    /**
     * Decreases the reference count
     * If the reference count becomes equal to zero, deletes the managed pointer
     */
    ~SharedPtr() { _remove(); }

    /**
     * Makes a copy of an input shared pointer and increments the reference count
     * \param[in] ptr   Shared pointer to copy
     */
    SharedPtr<T> & operator=(const SharedPtr<T> & ptr);

    /**
     * Releases managed pointer
     */
    void reset()
    {
        _remove();
        _ownedPtr = NULL;
        _refCount = NULL;
        _ptr      = NULL;
    }

    /**
     * Releases managed pointer, takes an ownership of ptr with default deleter
     * \tparam U    Class of the managed object
     * \param[in] ptr       Pointer to the managed object
     */
    template <class U>
    void reset(U * ptr);

    /**
     * Releases managed pointer, takes an ownership of ptr with deleter D
     * \tparam U    Class of the managed object
     * \tparam D    Class of the deleter object
     * \param[in] ptr       Pointer to the managed object
     * \param[in] deleter   Object used to delete the pointer when the reference count becomes equal to zero
     */
    template <class U, class D>
    void reset(U * ptr, const D & deleter);

    /**
     * Makes a copy of an input shared pointer and increments the reference count
     * \param[in] ptr   Shared pointer to copy
     */
    template <class U>
    SharedPtr<T> & operator=(const SharedPtr<U> & ptr)
    {
        if (((void *)&ptr != (void *)this) && ((void *)(ptr.get()) != (void *)(this->_ownedPtr)))
        {
            _remove();
            _ownedPtr = ptr._ownedPtr;
            _refCount = ptr._refCount;
            _ptr      = ptr._ptr;
            if (_refCount) _refCount->inc();
        }
        return *this;
    }

    /**
     * Dereferences a pointer to a managed object
     * \return  Pointer to the managed object
     */
    T * operator->() const { return _ptr; }

    /**
     * Dereferences a pointer to a managed object
     * \return  Reference to the managed object
     */
    T & operator*() const { return *_ptr; }

    /**
     * Checks if the managed pointer is not null
     * \return  true if the managed pointer is not null
     */
    operator bool() const { return (_ptr != NULL); }

    /**
     * Returns a pointer to a managed object
     * \return Pointer to the managed object
     */
    T * get() const { return _ptr; }

    /**
     * Returns a pointer to the beginning of owned memory
     * \return Pointer to the beginning of owned memory
     */
    T * getStartPtr() const { return _ownedPtr; }

    /**
     * Returns the number of shared_ptr objects referring to the same managed object
     * \return The number of shared_ptr objects referring to the same managed object
     */
    int useCount() const { return _refCount ? _refCount->get() : 0; }

protected:
    T * _ownedPtr;          /* Pointer to the beginning of the owned memory */
    T * _ptr;               /* Pointer to return */
    RefCounter * _refCount; /* Reference count */

    /**
    * Decreases the reference count
    * If the reference count becomes equal to zero, deletes the owned pointer
    */
    void _remove();

    template <class U>
    friend class SharedPtr;
}; // class SharedPtr

template <class T>
template <class U>
SharedPtr<T>::SharedPtr(const SharedPtr<U> & other) : _ownedPtr(other._ownedPtr), _ptr(other._ptr), _refCount(other._refCount)
{
    if (_refCount) _refCount->inc();
}

template <class T>
SharedPtr<T>::SharedPtr(const SharedPtr<T> & other) : _ownedPtr(other._ownedPtr), _ptr(other._ptr), _refCount(other._refCount)
{
    if (_refCount) _refCount->inc();
}

template <class T>
SharedPtr<T>::SharedPtr(const SharedPtr<T> & other, T * shiftedPtr) : _ownedPtr(other._ownedPtr), _ptr(shiftedPtr), _refCount(other._refCount)
{
    if (_refCount) _refCount->inc();
}

template <class T>
template <class U>
SharedPtr<T>::SharedPtr(const SharedPtr<U> & other, T * ptr, T * shiftedPtr) : _ownedPtr(ptr), _ptr(shiftedPtr), _refCount(other._refCount)
{
    if (_refCount) _refCount->inc();
}

template <class T>
void SharedPtr<T>::_remove()
{
    if (_refCount && (_refCount->dec() <= 0))
    {
        (*_refCount)(_ownedPtr);
        delete _refCount;
        _refCount = NULL;
        _ptr      = NULL;
    }
}

template <class T>
SharedPtr<T> & SharedPtr<T>::operator=(const SharedPtr<T> & ptr)
{
    if (&ptr == this) return *this;

    if (ptr._ownedPtr != this->_ownedPtr || ptr._ptr != this->_ptr)
    {
        _remove();
        _ownedPtr = ptr._ownedPtr;
        _refCount = ptr._refCount;
        _ptr      = ptr._ptr;
        if (_refCount) _refCount->inc();
    }
    return *this;
}

template <class T>
template <class U>
void SharedPtr<T>::reset(U * ptr)
{
    if (ptr != this->_ownedPtr)
    {
        _remove();
        _ownedPtr = ptr;
        _ptr      = ptr;
        _refCount = (ptr ? new RefCounterImp<ObjectDeleter<U> >() : NULL);
    }
}

template <class T>
template <class U, class D>
void SharedPtr<T>::reset(U * ptr, const D & deleter)
{
    if (ptr != this->_ownedPtr)
    {
        _remove();
        _ownedPtr = ptr;
        _ptr      = ptr;
        _refCount = (ptr ? new RefCounterImp<D>(deleter) : NULL);
    }
}

/**
 * Creates a new instance of SharedPtr whose managed object type is obtained from the type of the managed object of r
 * using a cast expression. Both shared pointers share ownership of the managed object.
 * The managed object of the resulting SharedPtr is obtained by calling static_cast<T*>(r.get()).
 */
template <class T, class U>
SharedPtr<T> staticPointerCast(const SharedPtr<U> & r)
{
    T * shifted = static_cast<T *>(r.get());
    T * start   = static_cast<T *>(r.getStartPtr());
    return SharedPtr<T>(r, start, shifted);
}

/**
 * Creates a new instance of SharedPtr whose managed object type is obtained from the type of the managed object of r
 * using a cast expression. Both shared pointers share ownership of the managed object.
 * The managed object of the resulting SharedPtr is obtained by calling reinterpret_cast<T*>(r.get()).
 */
template <class T, class U>
SharedPtr<T> reinterpretPointerCast(const SharedPtr<U> & r)
{
    T * shifted = reinterpret_cast<T *>(r.get());
    T * start   = reinterpret_cast<T *>(r.getStartPtr());
    return SharedPtr<T>(r, start, shifted);
}

/**
 * Creates a new instance of SharedPtr whose managed object type is obtained from the type of the managed object of r
 * using a cast expression. Both shared pointers share ownership of the managed object.
 * The managed object of the resulting SharedPtr is obtained by calling dynamic_cast<T*>(r.get()).
 */
template <class T, class U>
SharedPtr<T> dynamicPointerCast(const SharedPtr<U> & r)
{
    T * shifted = dynamic_cast<T *>(r.get());
    T * start   = dynamic_cast<T *>(r.getStartPtr());
    if (!r.get() || start)
    {
        return SharedPtr<T>(r, start, shifted);
    }
    else
    {
        return SharedPtr<T>();
    }
}
/** @} */
} // namespace interface1
using interface1::DeleterIface;
using interface1::ObjectDeleter;
using interface1::EmptyDeleter;
using interface1::ServiceDeleter;
using interface1::RefCounter;
using interface1::RefCounterImp;
using interface1::SharedPtr;
using interface1::staticPointerCast;
using interface1::dynamicPointerCast;
using interface1::reinterpretPointerCast;

} // namespace services
} // namespace daal

#endif
