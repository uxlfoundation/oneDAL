/* file: algorithm_types.h */
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
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __ALGORITHM_TYPES_H__
#define __ALGORITHM_TYPES_H__

#include "services/daal_defines.h"
#include "data_management/data/data_archive.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_collection.h"
#include "services/error_handling.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
/**
 * @defgroup base_algorithms Base Classes
 * \copydoc daal::algorithms
 * @ingroup algorithms
 */
/**
 * \brief Contains classes that implement algorithms for data analysis(data mining), and data modeling(training and prediction).
 *        These algorithms include matrix decompositions, clustering algorithms, classification and regression algorithms,
 *        as well as association rules discovery.
 */
namespace algorithms
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @ingroup base_algorithms
 * @{
 */
/**
 *  <a name="DAAL-STRUCT-ALGORITHMS__PARAMETER"></a>
 *  \brief %Base class to represent computation parameters.
 *         Algorithm-specific parameters are represented as derivative classes of the Parameter class.
 */
struct Parameter
{
    DAAL_NEW_DELETE();

    Parameter() {}
    virtual ~Parameter() {}

    virtual services::Status check() const { return services::Status(); }
};

/**
 * \brief Implements the abstract interface HyperparameterIface.
 *        Represents the common interface for performance-related hyperparameters of the computation.
 */
struct DAAL_EXPORT HyperparameterIface
{
    /**
     * Sets integer hyperparameter into this structure
     * \param[in] id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[in] value The value of the hyperparameter
     * \return Execution status
     */
    virtual services::Status set(unsigned int id, DAAL_INT64 value) = 0;

    /**
     * Sets double precision hyperparameter into this structure
     * \param[in] id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[in] value The value of the hyperparameter
     * \return Execution status
     */
    virtual services::Status set(unsigned int id, double value) = 0;

    /**
     * Finds integer hyperparameter in this structure
     * \param[in]  id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[out] value The value of the hyperparameter
     * \return Execution status.
     *         ErrorHyperparameterNotFound is returned if the 'id' of the hyperparameter cannot be foun
     *         in the structure.
     */
    virtual services::Status find(unsigned int id, DAAL_INT64 & value) const = 0;

    /**
     * Finds double precision hyperparameter in this structure
     * \param[in]  id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[out] value The value of the hyperparameter
     * \return Execution status.
     *         ErrorHyperparameterNotFound is returned if the 'id' of the hyperparameter cannot be foun
     *         in the structure.
     */
    virtual services::Status find(unsigned int id, double & value) const = 0;

    virtual ~HyperparameterIface() {}
};

/**
 * \brief %Base class to represent the implementation of performance-related hyperparameters
 *        of the computation.
 */
struct HyperparameterBaseImpl : public HyperparameterIface
{};

/**
 *  \brief %Base class to represent performance-related hyperparameters of the computation.
 *         Algorithm-specific hyperparameters are represented as derivative classes
 *         of the Hyperparameter class.
 */
struct DAAL_EXPORT Hyperparameter : protected HyperparameterIface
{
    DAAL_NEW_DELETE();

    /**
     * Constructs the requested number of performance-related hyperparameters for the algorithm
     * \param[in] intParamCount     Number of integer hyperparameters
     * \param[in] doubleParamCount  Number of double precision hyperparameters
     */
    Hyperparameter(size_t intParamCount = 0, size_t doubleParamCount = 0);

    virtual ~Hyperparameter() {}

protected:
    /**
     * Sets integer hyperparameter into this structure
     * \param[in] id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[in] value The value of the hyperparameter
     * \return Execution status
     */
    services::Status set(unsigned int id, DAAL_INT64 value) final;

    /**
     * Sets double precision hyperparameter into this structure
     * \param[in] id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[in] value The value of the hyperparameter
     * \return Execution status
     */
    services::Status set(unsigned int id, double value) final;

    /**
     * Finds integer hyperparameter in this structure
     * \param[in]  id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[out] value The value of the found hyperparameter
     * \return Execution status.
     *         ErrorHyperparameterNotFound is returned if the 'id' of the hyperparameter cannot be foun
     *         in the structure.
     */
    services::Status find(unsigned int id, DAAL_INT64 & value) const final;

    /**
     * Finds double precision hyperparameter in this structure
     * \param[in]  id    Unique to the particular algorithm identifier of the hyperparameter
     * \param[out] value The value of the found hyperparameter
     * \return Execution status.
     *         ErrorHyperparameterNotFound is returned if the 'id' of the hyperparameter cannot be foun
     *         in the structure.
     */
    services::Status find(unsigned int id, double & value) const final;

    /** Pointer to the implementation */
    services::SharedPtr<HyperparameterBaseImpl> _pimpl;
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__ARGUMENT"></a>
 *  \brief %Base class to represent computation input and output arguments.
 */
class DAAL_EXPORT Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty arguments */
    Argument() : idx(0) {}

    /**
     * Constructs conputation argument that contains n elements
     * \param[in] n     Number of elements in the argument
     */
    Argument(const size_t n);

    virtual ~Argument() {};

    /**
     * Inserts element into this argument structure
     * \param[in] val   Element to insert
     * \return Updated argument structure
     */
    Argument & operator<<(const data_management::SerializationIfacePtr & val)
    {
        (*_storage) << val;
        return *this;
    }

    /**
     * Retrieves number of elements in the argument
     * \return  Number of elements in the argument
     */
    size_t size() const { return _storage->size(); }

protected:
    /**
     * Copy constructor
     * \param[in] other Instance of the same class to copy
     */
    Argument(const Argument & other);

    /**
     * Copy assignment operator
     * \param[in] other Instance of the same class to copy
     */
    Argument & operator=(const Argument & other);

    /**
     * Retrieves specified element
     * \param[in] index Index of the element
     * \return Reference to the requested element
     */
    const data_management::SerializationIfacePtr & get(size_t index) const;

    /**
     * Sets the element to the specified position in the Argument
     * \param[in] index Index of the element
     * \param[in] value Pointer to the element
     * \return Reference to the requested element
     */
    void set(size_t index, const data_management::SerializationIfacePtr & value);

    /**
    * Sets the custom storage in the Argument
    * \param[in] storage custom defined storage
    */
    void setStorage(const data_management::DataCollectionPtr & storage);

    /**
    * Gets the storage in the Argument
    * \return Storage
    */
    static data_management::DataCollectionPtr & getStorage(Argument & a);

    /**
    * Gets the const storage in the Argument
    * \return Storage
    */
    static const data_management::DataCollectionPtr & getStorage(const Argument & a);

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(idx);
        arch->setObj(_storage.get());

        return services::Status();
    }

private:
    size_t idx;
    data_management::DataCollectionPtr _storage;
};

/**
*  <a name="DAAL-CLASS-ALGORITHMS__SERIALIZABLE_ARGUMENT"></a>
*  \brief %Base class to represent argument with serialization methods
*/
class SerializableArgument : public data_management::SerializationIface, public Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty argument */
    SerializableArgument() {}

    /**
    * Constructs argument with n elements
    * \param[in] n     Number of elements in the input argument
    */
    SerializableArgument(const size_t n) : Argument(n) {}

    virtual ~SerializableArgument() {};
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__INPUT"></a>
 *  \brief %Base class to represent computation input arguments.
 *         Algorithm-specific input arguments are represented as derivative classes of the Input class.
 */
class Input : public Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty input arguments */
    Input() {}

    /**
     * Constructs input arguments with n elements
     * \param[in] n     Number of elements in the input argument
     */
    Input(const size_t n) : Argument(n) {}

    virtual ~Input() {};

    /**
     * Checks the correctness of the input arguments
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method        Computation method
     */
    virtual services::Status check(const Parameter * /*parameter*/, int /*method*/) const { return services::Status(); }

protected:
    /**
    * Copy constructor
    * \param[in] other Instance of the same class to copy
    */
    Input(const Input & other) : Argument(other) {}

    Input & operator=(const Input & other) = default;
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PARTIALRESULT"></a>
 *  \brief %Base class to represent partial results of the computation.
 *         Algorithm-specific partial results are represented as derivative classes of the PartialResult class.
 */
class PartialResult : public SerializableArgument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty partial results */
    PartialResult() : _initFlag(false) {};

    /**
     * Constructs partial results with n elements
     * \param[in] n     Number of elements in the partial results
     */
    PartialResult(const size_t n) : SerializableArgument(n), _initFlag(false) {}

    virtual ~PartialResult() {};

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    DAAL_EXPORT virtual int getSerializationTag() const { return 0; }

    /**
     * Retrieves the initialization flag
     * \return Initialization flag. True, if the partial results are already initialized; false - otherwise.
     */
    bool getInitFlag() { return _initFlag; }

    /**
     * Sets the initialization flag
     * \param[in] flag  Initialization flag. True, if the partial results are already initialized; false - otherwise.
     */
    void setInitFlag(bool flag) { _initFlag = flag; }

    /**
     * Checks the correctness of the partial results structure
     * \param[in] input         Pointer to the input arguments of the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method        Computation method
     */
    virtual services::Status check(const Input * /*input*/, const Parameter * /*parameter*/, int /*method*/) const { return services::Status(); }

    /**
    * Checks the correctness of the partial results structure
    * \param[in] parameter     Pointer to the parameters of the algorithm
    * \param[in] method        Computation method
    */
    virtual services::Status check(const Parameter * /*parameter*/, int /*method*/) const { return services::Status(); }

private:
    bool _initFlag;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        Argument::serialImpl<Archive, onDeserialize>(arch);

        arch->set(_initFlag);

        return services::Status();
    }
    virtual services::Status serializeImpl(data_management::InputDataArchive * /*archive*/) { return services::Status(); }
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * /*archive*/) { return services::Status(); }
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__RESULT"></a>
 *  \brief %Base class to represent final results of the computation.
 *         Algorithm-specific final results are represented as derivative classes of the Result class.
 */
class Result : public SerializableArgument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty final results */
    Result() {}

    /**
     * Constructs final results with n elements
     * \param[in] n     Number of elements in the final results
     */
    Result(const size_t n) : SerializableArgument(n) {}

    virtual ~Result() {};

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    DAAL_EXPORT virtual int getSerializationTag() const { return 0; }

    /**
     * Checks the correctness of the final results structure
     * \param[in] input         Pointer to the input arguments of the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method        Computation method
     */
    virtual services::Status check(const Input * /*input*/, const Parameter * /*parameter*/, int /*method*/) const { return services::Status(); }

    /**
    * Checks the correctness of the partial result structure
    * \param[in] partialResult Pointer to the partial result arguments of the algorithm
    * \param[in] parameter     Pointer to the parameters of the algorithm
    * \param[in] method        Computation method
    */
    virtual services::Status check(const PartialResult * /*partialResult*/, const Parameter * /*parameter*/, int /*method*/) const
    {
        return services::Status();
    }

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        Argument::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }
    virtual services::Status serializeImpl(data_management::InputDataArchive * /*archive*/) { return services::Status(); }
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * /*archive*/) { return services::Status(); }
};

/**
*  <a name="DAAL-CLASS-ALGORITHMS__OPTIONAL_ARGUMENT"></a>
*  \brief %Base class to represent argument with serialization methods
*/
class DAAL_EXPORT OptionalArgument : public SerializableArgument
{
public:
    DECLARE_SERIALIZABLE_TAG()

    /** Default constructor. Constructs empty argument */
    OptionalArgument() : SerializableArgument(0) {}

    /**
    * Constructs argument with n elements
    * \param[in] n     Number of elements in the input argument
    */
    OptionalArgument(const size_t n) : SerializableArgument(n) {}

    /**
    * Retrieves specified element
    * \param[in] index Index of the element
    * \return Reference to the requested element
    */
    const data_management::SerializationIfacePtr & get(size_t index) const { return SerializableArgument::get(index); }

    /**
    * Sets the element to the specified position in the Argument
    * \param[in] index Index of the element
    * \param[in] value Pointer to the element
    * \return Reference to the requested element
    */
    void set(size_t index, const data_management::SerializationIfacePtr & value) { return SerializableArgument::set(index, value); }

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        Argument::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }
};
typedef services::SharedPtr<Input> InputPtr;
typedef services::SharedPtr<PartialResult> PartialResultPtr;
typedef services::SharedPtr<Result> ResultPtr;
typedef services::SharedPtr<OptionalArgument> OptionalArgumentPtr;

/** @} */
} // namespace interface1
using interface1::HyperparameterBaseImpl;
using interface1::Hyperparameter;
using interface1::Parameter;
using interface1::Argument;
using interface1::Input;
using interface1::InputPtr;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::Result;
using interface1::ResultPtr;
using interface1::OptionalArgument;
using interface1::OptionalArgumentPtr;

} // namespace algorithms
} // namespace daal
#endif
