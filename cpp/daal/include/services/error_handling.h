/** file error_handling.h */
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
//  Handling errors in oneAPI Data Analytics Library (oneDAL).
//--
*/

#ifndef __ERROR_HANDLING__
#define __ERROR_HANDLING__

#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
    #include <exception>
#endif

#include "services/daal_string.h"
#include "services/error_indexes.h"
#include "services/error_id.h"
#include "services/collection.h"
#include "services/internal/status_to_error_id.h"

namespace daal
{
namespace services
{
namespace interface1
{
/**
 * @ingroup error_handling
 * @{
 */
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))

/**
 * <a name="DAAL-STRUCT-SERVICES__EXCEPTION"></a>
 * \brief Class that represents an exception
 */
struct DAAL_EXPORT Exception : std::exception
{
public:
    /**
     * Constructs an exception with a description
     * \param[in] description    Description of the exception
     */
    Exception(const char * description) : _description(description) {};

    /**
     * Return description of the exception
     * \return Description of the exception
     */
    virtual const char * what() const throw() { return _description.c_str(); };

    #ifndef cxx11
    /**
     * Destructor of an the exception
     */
    virtual ~Exception() throw() {}
    #endif

    /**
     * Returns exception with description
     * \return Exception with description
     */
    static Exception getException(const String & description)
    {
        String d(description);
        return Exception(d.c_str());
    }

    /**
    * Returns exception with description
    * \return Exception with description
    */
    static Exception getException(const char * description) { return Exception(description); }

private:
    String _description;
};

#endif

} // namespace interface1
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
using interface1::Exception;
#endif

namespace interface1
{
/**
 * <a name="DAAL-CLASS-SERVICES__ERROR"></a>
 * \brief Class that represents an error
 */
class DAAL_EXPORT Error
{
public:
    DAAL_NEW_DELETE();

    /**
     * Constructs an error from an identifier
     * \param[in] id    Identifier of the error
     */
    Error(const ErrorID id = NoErrorMessageFound);

    /**
     * Copy constructor. Constructs an error from a copy of the content of another error
     * \param[in] e    Another error to be used as a source with which to initialize the contents of this error
     */
    Error(const Error & e);

    /** Destructor */
    ~Error();

    /**
     * Returns a description of the error
     * \return Identifier of this error
     */
    ErrorID id() const { return _id; }

    /**
     * Sets an identifier of the error
     * \param[in] id    Identifier of the error
     */
    void setId(ErrorID id) { _id = id; }

    /**
     * Returns the word description of the error
     * \return Description of the error
     */
    const char * description() const;

    /**
     * Adds an integer detail into a collection of details associated with this error
     * \param[in] id    Identifier of the detail
     * \param[in] value Value of the detail
     * \return Reference to this object
     */
    Error & addIntDetail(ErrorDetailID id, int value);

    /**
     * Adds a floating-point detail into a collection of details associated with this error
     * \param[in] id    Identifier of the detail
     * \param[in] value Value of the detail
     * \return Reference to this object
     */
    Error & addDoubleDetail(ErrorDetailID id, double value);

    /**
     * Adds a string detail into a collection of details associated with this error
     * \param[in] id    Identifier of the detail
     * \param[in] value Value of the detail
     * \return Reference to this object
     */
    Error & addStringDetail(ErrorDetailID id, const String & value);

    /**
    * Access to error details list
    * \return pointer to the top of error details list
    */
    const ErrorDetail * details() const { return _details; }

    /**
     * Create an instance of the class with the given arguments
     * \return pointer to the new instance
     */
    static SharedPtr<Error> create(ErrorID id);

    static SharedPtr<Error> create(ErrorID id, ErrorDetailID det, int value);

    static SharedPtr<Error> create(ErrorID id, ErrorDetailID det, const String & value);

protected:
    /**
    * Adds a detail into a collection of details associated with this error.
    * This class becomes an owner of the detail
    * \param[in] detail The detail
    * \return Reference to this object
    */
    Error & addDetail(ErrorDetail * detail);

private:
    ErrorID _id;
    ErrorDetail * _details;

    Error & operator=(const Error &);
};
typedef SharedPtr<Error> ErrorPtr;

/**
 * <a name="DAAL-CLASS-SERVICES__KERNELERRORCOLLECTION"></a>
 * \brief Class that represents a kernel error collection (collection that cannot throw exceptions)
 */
class DAAL_EXPORT KernelErrorCollection : private Collection<SharedPtr<Error> >
{
public:
    DAAL_NEW_DELETE();

    typedef Collection<SharedPtr<Error> > super;

    /**
     * Constructs a kernel error collection
     */
    KernelErrorCollection() : _description(0) {}

    /**
     * Copy constructor of a kernel error collection
     * \param[in] other Kernel error collection that will be copied
     */
    KernelErrorCollection(const KernelErrorCollection & other);

    /**
     * Assignment operator of a kernel error collection
     * \param[in] other Kernel error collection that will be copied
     */
    KernelErrorCollection & operator=(const KernelErrorCollection & other);

    /**
     * Adds error id to kernel error collection
     * \param[in] id Identifier of error
     * \return    Pointer to the new error
     */
    Error & add(const ErrorID & id);

    /**
     * Adds error to kernel error collection
     * \param[in] e Error to be added to kernel error collection
     */
    void add(const ErrorPtr & e);

    /**
     * Adds error kernel collection
     * \param[in] e Error collection that will be added
     */
    void add(const services::SharedPtr<KernelErrorCollection> & e);

    /**
    * Adds error kernel collection
    * \param[in] e Error collection that will be added
    */
    void add(const KernelErrorCollection & e);

    /**
     * Returns true if kernel error collection is empty
     * \return True if kernel error collection is empty
     */
    bool isEmpty() const { return size() == 0; }

    /**
     *  Size of a collection
     *  \return Size of the collection
     */
    size_t size() const;

    /**
    *  Element access
    *  \param[in] index Index of an accessed element
    *  \return    Pointer to the element
    */
    Error * at(size_t index);

    /**
    *  Element access
    *  \param[in] index Index of an accessed element
    *  \return    Pointer to the element
    */
    const Error * at(size_t index) const;

    /**
    *  Element access
    *  \param[in] index Index of an accessed element
    *  \return    Pointer to the element
    */
    Error * operator[](size_t index);

    /**
    *  Element access
    *  \param[in] index Index of an accessed element
    *  \return    Pointer to the element
    */
    const Error * operator[](size_t index) const;

    /**
     * Destructor of a kernel error collection
     */
    virtual ~KernelErrorCollection();

    /**
     * Returns description of all errors from collection
     * \return Description of all errors from collection
     */
    const char * getDescription() const;

private:
    mutable char * _description;
};
typedef SharedPtr<KernelErrorCollection> KernelErrorCollectionPtr;

/**
 * <a name="DAAL-CLASS-SERVICES__ERRORCOLLECTION"></a>
 * \brief Class that represents an error collection
 */
class DAAL_EXPORT ErrorCollection
{
public:
    DAAL_NEW_DELETE();

    /**
     * Constructs an error collection
     */
    ErrorCollection() : _errors(new KernelErrorCollection()), _canThrow(true) {}

    /**
     * Copy constructor for error collection
     * \param[in] o  Error collection that will be copied
     */
    ErrorCollection(const ErrorCollection & o) : _errors(o.getErrors()), _canThrow(o._canThrow) {}

    ErrorCollection & operator=(const ErrorCollection & o)
    {
        if (this != &o)
        {
            _errors   = o.getErrors();
            _canThrow = o._canThrow;
        }
        return *this;
    }

    /**
    * Constructor for error collection
    * \param[in] errors  Error collection that will be kept by
    */
    explicit ErrorCollection(const KernelErrorCollection & errors) : _errors(new KernelErrorCollection(errors)), _canThrow(true) {}

    /**
     * Adds error to error collection and throws exception if DAAL_NOTHROW_EXCEPTIONS not defined
     * \param[in] id Error identifier
     */
    void add(const ErrorID & id)
    {
        _errors->add(id);
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        if (_canThrow) throw Exception::getException(getDescription());
#endif
    }

    /**
     * Adds error to error collection and throws exception if DAAL_NOTHROW_EXCEPTIONS not defined
     * \param[in] e Error
     */
    void add(const ErrorPtr & e)
    {
        _errors->add(e);
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        if (_canThrow) throw Exception::getException(getDescription());
#endif
    }

    /**
     * Adds error collection to another error collection and throw exception if DAAL_NOTHROW_EXCEPTIONS not defined
     * \param[in] e Error collection
     */
    void add(const ErrorCollection & e)
    {
        if (!e.isEmpty())
        {
            _errors->add(e.getErrors());
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            if (_canThrow) throw Exception::getException(getDescription());
#endif
        }
    }

    /**
    * Adds kernel error collection to another error collection and throw exception if DAAL_NOTHROW_EXCEPTIONS not defined
    * \param[in] e Error collection
    */
    void add(const KernelErrorCollectionPtr & e)
    {
        if (!e->isEmpty())
        {
            _errors->add(e);
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            if (_canThrow) throw Exception::getException(getDescription());
#endif
        }
    }

    /**
     * Returns size of an error collection
     * \return Size of an error collection
     */
    size_t size() const
    {
        return _errors->size();
    }

    /**
     * Returns true if kernel error collection is empty
     * \return True if kernel error collection is empty
     */
    bool isEmpty() const
    {
        return _errors->isEmpty();
    }

    /**
     * Destructor of error collection
     */
    virtual ~ErrorCollection() {}

    /**
     * Returns kernel error collection
     * \return True if kernel error collection is empty
     */
    const KernelErrorCollectionPtr & getErrors() const
    {
        return _errors;
    }

    /**
     * Returns description of all errors from collection
     * \return Description of all errors from collection
     */
    const char * getDescription() const
    {
        return _errors->getDescription();
    }

    /**
    * Returns true if can throw exceptions
    * \return true if can throw exceptions
    */
    bool canThrow() const
    {
        return _canThrow;
    }

    /**
    * Set canThrow flag
    * \param[in] bOn The flag value
    * \return previous value
    */
    bool setCanThrow(bool bOn)
    {
        bool bVal = _canThrow;
        _canThrow = bOn;
        return bVal;
    }

private:
    KernelErrorCollectionPtr _errors;
    bool _canThrow;
};
typedef SharedPtr<ErrorCollection> ErrorCollectionPtr;
/** @} */

/**
* <a name="DAAL-CLASS-SERVICES__ERRORCODE"></a>
* \brief Class that holds the results of API calls. In case of API routine failure it contains the
*        list of errors describing problems API encountered.
*/
class DAAL_EXPORT Status
{
public:
    friend ErrorID daal::services::internal::get_error_id(const Status &);
    /**
     * Default constructor
     */
    Status() : _impl(0) {}
    /**
    * Constructs an object containing given error
    * \param[in] id Identifier of the error
    */
    Status(ErrorID id);
    /**
    * Constructs an object containing given error
    * \param[in] e the error
    */
    Status(const ErrorPtr & e);

    /**
    * Copy constructor
    */
    Status(const Status & other);

    /**
    * Destructor
    */
    ~Status();

    /**
    * Checks that the object doesn't contain errors
    * \return true if no errors false otherwise
    */
    bool ok() const { return !_impl; }

    /**
    * Checks that the object doesn't contain errors
    * \return true if no errors false otherwise
    */
    operator bool() const { return ok(); }

    /**
    * Adds an error to the object
    * \param[in] id    Identifier of the error
    * \return Reference to this object
    */
    Status & add(ErrorID id);

    /**
    * Adds an error to the object
    * \param[in] e the error
    * \return Reference to this object
    */
    Status & add(const ErrorPtr & e);

    /**
    * Adds another status to this object
    * \param[in] other the status to add
    * \return Reference to this object
    */
    Status & add(const Status & other);

    /**
    * Operator, adds another status to this object
    * \param[in] other the status to add
    * \return Reference to this object
    */
    Status & operator|=(const Status & other) { return add(other); }

    /**
    * Assignment oerator
    * \param[in] other the status to assign
    * \return Reference to this object
    */
    Status & operator=(const Status & other);

    /**
    * Returns description of all errors contained in the object
    * \return Description of all errors contained in the object
    */
    const char * getDescription() const;

    /**
    * Clear all errors contained in this object
    * \return Description of all errors contained in the object
    */
    void clear();

    /**
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED Status(const KernelErrorCollection & e);
    /**
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED Status(const ErrorCollection & e);

    /**
     * \DAAL_DEPRECATED
     */
    ErrorCollectionPtr getCollection() const;

private:
    void * _impl; //pointer to the collection of errors with reference counting
};

inline const Status & throwIfPossible(const Status & s)
{
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
    if (!s.ok()) throw services::Exception::getException(s.getDescription());
#endif
    return s;
}

} // namespace interface1
using interface1::Error;
using interface1::KernelErrorCollection;
using interface1::ErrorCollection;
using interface1::ErrorPtr;
using interface1::KernelErrorCollectionPtr;
using interface1::ErrorCollectionPtr;
using interface1::Status;
using interface1::throwIfPossible;
namespace internal
{
inline Status checkForNullByteInjection(const char * begin, const char * const end)
{
    for (; begin != end; ++begin)
    {
        if (*begin == '\0')
        {
            return Status(ErrorNullByteInjection);
        }
    }
    return Status();
}
} // namespace internal
} // namespace services
} // namespace daal
#endif
