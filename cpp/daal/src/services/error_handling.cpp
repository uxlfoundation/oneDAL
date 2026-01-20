/** file error_handling.cpp */
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

#include "services/error_handling.h"
#include "data_management/features/defines.h"

#include "src/threading/threading.h"
#include "src/services/service_string_utils.h"
#include "src/algorithms/service_error_handling.h"

namespace daal
{
namespace services
{
/*
    Example:

    SharedPtr<Error> e(new Error(ErrorIncorrectNumberOfFeatures));
    e->addIntDetail(Row, 10);
    e->addIntDetail(Column, 40);
    e->addStringDetail(services::Method, services::String("CorrelationDense"));
    this->_errors->add(e);
    this->_errors->add(services::ErrorIncorrectNumberOfObservations);
    this->_errors->add(services::ErrorIncorrectNumberOfElementsInResultCollection);
*/

namespace
{
/**
* <a name="DAAL-ENUM-SERVICES__MESSAGE"></a>
* \brief Class that represents Message
* \tparam IDType Type of message
*/
template <class IDType>
class Message
{
public:
    /**
    * Constructs Message from identifier and description
    * \param[in] id Error identifier
    * \param[in] description Description for message
    */
    Message(const IDType & id, const char * description) : _id(id), _description(description) {};

    /**
    * Destructor of Message class
    */
    virtual ~Message() {};

    /**
    * Returns identifier of a message
    * \return identifier of a message
    */
    const IDType id() const { return _id; }

    /**
    * Returns description of a message
    * \return description of a message
    */
    const char * description() const { return _description.c_str(); }

private:
    IDType _id;
    const String _description;
};

/**
* <a name="DAAL-ENUM-SERVICES__MESSAGECOLLECTION"></a>
* \brief Class that represents an Message collection
* \tparam IDType Type of message in message collection
*/
template <class IDType>
class MessageCollection : public Collection<SharedPtr<Message<IDType> > >
{
public:
    /**
    * Constructs a message collection
    * \param[in] noMessageFound Index returned when no corresponding element is found in the collection
    */
    MessageCollection(IDType noMessageFound) : _noMessageFound(noMessageFound) {};

    /**
    * Finds message for error by error ID
    * \param[in] id Error identifier
    * \return Pointer to message
    */
    services::SharedPtr<Message<IDType> > find(IDType id) const
    {
        bool found   = false;
        size_t index = 0;

        if (this->size() == 0)
        {
            return find(_noMessageFound);
        }

        for (size_t i = 0; i < this->size() && found == false; i++)
        {
            if ((*this)[i]->id() == id)
            {
                found = true;
                index = i;
            }
        }

        if (found)
        {
            return (*this)[index];
        }
        else
        {
            return find(_noMessageFound);
        }
    }

    /**
    * Destructor of a message collection
    */
    virtual ~MessageCollection() {};

protected:
    const IDType _noMessageFound;
};

/**
* <a name="DAAL-ENUM-SERVICES__ERRORMESSAGECOLLECTION"></a>
* \brief Class that represents an error message collection
*/
class ErrorMessageCollection : public MessageCollection<ErrorID>
{
public:
    /**
    * Constructs an error message collection
    */
    ErrorMessageCollection() : MessageCollection<ErrorID>(NoErrorMessageFound) { parseResourceFile(); }

    /**
    * Destructor of an error message collection
    */
    ~ErrorMessageCollection() {}

    void add(ErrorID id, const char * value) { push_back(services::SharedPtr<Message<ErrorID> >(new Message<ErrorID>(id, value))); }

protected:
    void parseResourceFile();
};

/**
* <a name="DAAL-ENUM-SERVICES__ERRORDETAILCOLLECTION"></a>
* \brief Class that represents an error detail collection
*/
class ErrorDetailCollection : public MessageCollection<ErrorDetailID>
{
public:
    /**
    * Construct an error detail collection
    */
    ErrorDetailCollection() : MessageCollection<ErrorDetailID>(NoErrorMessageDetailFound) { parseResourceFile(); }

    /**
    * Destructor of an error detail collection
    */
    ~ErrorDetailCollection() {}

    void add(ErrorDetailID id, const char * value) { push_back(services::SharedPtr<Message<ErrorDetailID> >(new Message<ErrorDetailID>(id, value))); }

protected:
    void parseResourceFile();
};

const ErrorMessageCollection & errorMessageCollection()
{
    static const ErrorMessageCollection inst;
    return inst;
}

const ErrorDetailCollection & errorDetailCollection()
{
    static const ErrorDetailCollection inst;
    return inst;
}

int cat(const char * source, char * destination)
{
    return daal::internal::ServiceInst::serv_strncat_s(
        destination, DAAL_MAX_STRING_SIZE, source,
        DAAL_MAX_STRING_SIZE - daal::internal::ServiceInst::serv_strnlen_s(destination, DAAL_MAX_STRING_SIZE));
}
} // namespace

/**
* <a name="DAAL-ENUM-SERVICES__ERRORDETAILIMPL"></a>
* \brief Class that implements error detail interface
* \tparam Type of value in an error detail
*/
template <typename T>
class ErrorDetailImpl : public ErrorDetail
{
public:
    DAAL_NEW_DELETE();

    /**
    * Constructs error detail from error identifier and value
    * \param[in] id    Error identifier
    * \param[in] value Value of error detail
    */
    ErrorDetailImpl(ErrorDetailID id, const T & value) : ErrorDetail(id), _value(value) {}

    /**
    * Destructor
    */
    ~ErrorDetailImpl() {}

    /**
    * Returns value of an error detail
    * \return value of an error detail
    */
    T value() const { return _value; }

    /**
    * Returns copy of this object
    * \return copy of this object
    */
    virtual ErrorDetail * clone() const { return new ErrorDetailImpl<T>(id(), value()); }

    /**
    * Adds description of the error detail to the given string
    * \param[in] str String to add description to
    */
    virtual void describe(char * str) const;

private:
    const T _value;
};

template <typename T>
void ErrorDetailImpl<T>::describe(char * str) const
{
    daal::internal::ServiceInst::serv_strncat_s(str, DAAL_MAX_STRING_SIZE, errorDetailCollection().find(id())->description(),
                                                DAAL_MAX_STRING_SIZE - daal::internal::ServiceInst::serv_strnlen_s(str, DAAL_MAX_STRING_SIZE));
    daal::internal::ServiceInst::serv_strncat_s(str, DAAL_MAX_STRING_SIZE, ": ",
                                                DAAL_MAX_STRING_SIZE - daal::internal::ServiceInst::serv_strnlen_s(str, DAAL_MAX_STRING_SIZE));
    char buffer[DAAL_MAX_STRING_SIZE] = { 0 };
    internal::toStringBuffer<T>(value(), buffer);
    daal::internal::ServiceInst::serv_strncat_s(str, DAAL_MAX_STRING_SIZE, buffer,
                                                DAAL_MAX_STRING_SIZE - daal::internal::ServiceInst::serv_strnlen_s(str, DAAL_MAX_STRING_SIZE));

    daal::internal::ServiceInst::serv_strncat_s(str, DAAL_MAX_STRING_SIZE, "\n",
                                                DAAL_MAX_STRING_SIZE - daal::internal::ServiceInst::serv_strnlen_s(str, DAAL_MAX_STRING_SIZE));
}

Error::Error(const ErrorID id) : _id(id), _details(nullptr) {}

Error::Error(const Error & e) : _id(e._id), _details(nullptr)
{
    ErrorDetail * pCur = nullptr;
    for (auto ptr = e.details(); ptr; ptr = ptr->next())
    {
        auto pClone = ptr->clone();
        if (pCur)
        {
            pCur->addNext(pClone);
            pCur = pClone;
        }
        else
        {
            _details = pCur = pClone;
        }
    }
}

Error::~Error()
{
    for (auto ptr = _details; ptr;)
    {
        auto tmp = ptr->next();
        delete ptr;
        ptr = tmp;
    }
}

const char * Error::description() const
{
    return errorMessageCollection().find(_id)->description();
}

Error & Error::addDetail(ErrorDetail * detail)
{
    if (detail)
    {
        auto ptr = _details;
        if (ptr)
        {
            for (; ptr->next(); ptr = ptr->next())
                ;
            ptr->addNext(detail);
        }
        else
            _details = detail;
    }
    return *this;
}

Error & Error::addIntDetail(ErrorDetailID id, int value)
{
    return addDetail(new ErrorDetailImpl<int>(id, value));
}

Error & Error::addDoubleDetail(ErrorDetailID id, double value)
{
    return addDetail(new ErrorDetailImpl<double>(id, value));
}

Error & Error::addStringDetail(ErrorDetailID id, const String & value)
{
    return addDetail(new ErrorDetailImpl<String>(id, value));
}

ErrorPtr Error::create(ErrorID id)
{
    return ErrorPtr(new Error(id));
}

ErrorPtr Error::create(ErrorID id, ErrorDetailID det, int value)
{
    ErrorPtr ptr(new Error(id));
    ptr->addIntDetail(det, value);
    return ptr;
}

ErrorPtr Error::create(ErrorID id, ErrorDetailID det, const String & value)
{
    ErrorPtr ptr(new Error(id));
    ptr->addStringDetail(det, value);
    return ptr;
}

KernelErrorCollection::KernelErrorCollection(const KernelErrorCollection & other) : super(other), _description(nullptr) {}

KernelErrorCollection & KernelErrorCollection::operator=(const KernelErrorCollection & other)
{
    if (&other != this)
    {
        super::operator=(other);
        _description = nullptr;
    }

    return *this;
}

Error & KernelErrorCollection::add(const ErrorID & id)
{
    if (size())
    {
        auto pLast = (*this)[size() - 1];
        if (pLast && (pLast->id() == id) && !pLast->details()) return *pLast; //no error duplication in the collection
    }
    ErrorPtr p(new Error(id));
    push_back(p);
    return *p.get();
}

void KernelErrorCollection::add(const ErrorPtr & e)
{
    if (e.get() && !e->details())
    {
        if (size())
        {
            auto pLast = (*this)[size() - 1];
            if (pLast && (pLast->id() == e->id()) && !pLast->details()) return; //no error duplication in the collection
        }
    }
    push_back(e);
}

void KernelErrorCollection::add(const services::SharedPtr<KernelErrorCollection> & e)
{
    add(*e);
}

void KernelErrorCollection::add(const KernelErrorCollection & o)
{
    const super & c = o;
    size_t i        = 0;
    if (size() && o.size())
    {
        auto pLast = (*this)[size() - 1];
        if (pLast && !pLast->details())
        {
            auto pFirst = o[0];
            if (pFirst && !pFirst->details() && (pLast->id() == pFirst->id())) ++i; //no error duplication in the collection
        }
    }
    for (; i < o.size(); i++) push_back(c[i]);
}

KernelErrorCollection::~KernelErrorCollection()
{
    if (_description)
    {
        daal_free(_description);
        _description = NULL;
    }
}

const char * KernelErrorCollection::getDescription() const
{
    if (size() == 0)
    {
        if (_description)
        {
            daal_free(_description);
        }
        _description    = (char *)daal::services::daal_calloc(sizeof(char) * 1);
        _description[0] = '\0';
        return _description;
    }

    size_t descriptionSize   = 0;
    char ** errorDescription = (char **)daal::services::daal_calloc(sizeof(char *) * size());

    for (size_t i = 0; i < size(); i++)
    {
        errorDescription[i]    = (char *)daal::services::daal_calloc(sizeof(char) * (DAAL_MAX_STRING_SIZE));
        errorDescription[i][0] = '\0';

        services::SharedPtr<Error> e = _array[i];

        const char * currentDescription = errorMessageCollection().find(e->id())->description();
        cat(currentDescription, errorDescription[i]);

        const char * newLine = "\n";
        cat(newLine, errorDescription[i]);

        if (e->details())
        {
            const char * details = "Details:\n";
            cat(details, errorDescription[i]);
            for (const auto * ptr = e->details(); ptr; ptr = ptr->next()) ptr->describe(errorDescription[i]);
        }

        descriptionSize += daal::internal::ServiceInst::serv_strnlen_s(errorDescription[i], DAAL_MAX_STRING_SIZE);
    }

    if (_description)
    {
        daal_free(_description);
    }
    _description    = (char *)daal::services::daal_calloc(sizeof(char) * (descriptionSize + 1));
    _description[0] = '\0';

    for (size_t i = 0; i < size(); i++)
    {
        cat(errorDescription[i], _description);
        daal_free(errorDescription[i]);
        errorDescription[i] = NULL;
    }

    daal_free(errorDescription);
    errorDescription = NULL;
    return _description;
}

size_t KernelErrorCollection::size() const
{
    return super::size();
}

Error * KernelErrorCollection::at(size_t index)
{
    return super::operator[](index).get();
}

const Error * KernelErrorCollection::at(size_t index) const
{
    return super::operator[](index).get();
}

Error * KernelErrorCollection::operator[](size_t index)
{
    return super::operator[](index).get();
}

const Error * KernelErrorCollection::operator[](size_t index) const
{
    return super::operator[](index).get();
}

/////////////////////////////////////////////////////////////////////////////////////////
// Status
/////////////////////////////////////////////////////////////////////////////////////////
class SharedErrorCollection : public KernelErrorCollection
{
public:
    SharedErrorCollection() : _nRefs(1) {}
    SharedErrorCollection(const SharedErrorCollection & o) : KernelErrorCollection(o), _nRefs(1) {}
    SharedErrorCollection(const KernelErrorCollection & o) : KernelErrorCollection(o), _nRefs(1) {}
    void addRef() { _nRefs.inc(); }
    void release()
    {
        if (!_nRefs.dec()) delete this;
    }
    int numRefs() const { return _nRefs.get(); }

private:
    AtomicInt _nRefs;
};

Status::Status(ErrorID id) : _impl(nullptr)
{
    add(id);
}

Status::Status(const ErrorPtr & e) : _impl(nullptr)
{
    add(e);
}

Status::Status(const Status & other) : _impl(other._impl)
{
    if (_impl) static_cast<SharedErrorCollection *>(_impl)->addRef();
}

Status & Status::operator=(const Status & other)
{
    if (_impl) static_cast<SharedErrorCollection *>(_impl)->release();
    _impl = other._impl;
    if (_impl) static_cast<SharedErrorCollection *>(_impl)->addRef();
    return *this;
}

Status::Status(const KernelErrorCollection & e) : _impl(nullptr)
{
    if (e.size()) _impl = new SharedErrorCollection(e);
}

Status::Status(const ErrorCollection & e) : _impl(nullptr)
{
    if (e.size()) _impl = new SharedErrorCollection(*e.getErrors());
}

Status::~Status()
{
    if (_impl) static_cast<SharedErrorCollection *>(_impl)->release();
}

void Status::clear()
{
    if (_impl)
    {
        static_cast<SharedErrorCollection *>(_impl)->release();
        _impl = nullptr;
    }
}

static SharedErrorCollection * getOwnCopy(void *& ptr)
{
    SharedErrorCollection * res = static_cast<SharedErrorCollection *>(ptr);
    if (res)
    {
        if (res->numRefs() == 1) return res;
        res = new SharedErrorCollection(*res);
    }
    else
    {
        res = new SharedErrorCollection();
    }
    ptr = res;
    return res;
}

Status & Status::add(ErrorID id)
{
    getOwnCopy(_impl)->add(id);
    return *this;
}

Status & Status::add(const ErrorPtr & e)
{
    getOwnCopy(_impl)->add(e);
    return *this;
}

Status & Status::add(const Status & other)
{
    if (other._impl)
    {
        if (_impl)
            getOwnCopy(_impl)->add(*static_cast<SharedErrorCollection *>(other._impl));
        else
        {
            _impl = other._impl;
            static_cast<SharedErrorCollection *>(_impl)->addRef();
        }
    }
    return *this;
}

const char * Status::getDescription() const
{
    return _impl ? static_cast<SharedErrorCollection *>(_impl)->getDescription() : nullptr;
}

services::ErrorCollectionPtr Status::getCollection() const
{
    if (_impl)
    {
        const SharedErrorCollection & impl = *static_cast<const SharedErrorCollection *>(_impl);
        return services::ErrorCollectionPtr(new services::ErrorCollection(impl));
    }
    return services::ErrorCollectionPtr(new services::ErrorCollection());
}

namespace internal
{
ErrorID get_error_id(const Status & s)
{
    if (s._impl)
    {
        return static_cast<SharedErrorCollection *>(s._impl)->at(0)->id();
    }
    return ErrorID::UnknownError;
}

} // namespace internal

/////////////////////////////////////////////////////////////////////////////////////////

void ErrorMessageCollection::parseResourceFile()
{
    // Input errors: -1..-1999
    add(ErrorIncorrectElementInNumericTableCollection, "Incorrect element in collection of numeric tables");
    add(ErrorIncorrectElementInPartialResultCollection, "Incorrect element in collection of partial results");
    add(ErrorNullPartialResultDataCollection, "Null partial result data collection");
    add(ErrorIncorrectElementInCollection, "Incorrect element in collection");
    add(ErrorEmptyAuxiliaryDataCollection, "Empty auxiliary data collection");
    add(ErrorNullAuxiliaryDataCollection, "Null auxiliary data collection");
    add(ErrorNullAuxiliaryAlgorithm, "Null auxiliary algorithm");
    add(ErrorNullInitializationProcedure, "Null initialization procedure");
    add(ErrorNumericTableIsNotSquare, "Numeric table is not square");
    add(ErrorNullNumericTable, "Null numeric table is not supported");
    add(ErrorIncorrectNumberOfColumns, "Number of columns in numeric table is incorrect");
    add(ErrorIncorrectNumberOfRows, "Number of rows in numeric table is incorrect");
    add(ErrorIncorrectTypeOfNumericTable, "Incorrect type of Numeric Table");
    add(ErrorUnsupportedCSRIndexing, "CSR Numeric Table has unsupported indexing type");
    add(ErrorSignificanceLevel, "Incorrect significance level value");
    add(ErrorAccuracyThreshold, "Incorrect accuracy threshold");
    add(ErrorIncorrectNumberOfBetas, "Incorrect number of betas in linear regression model");
    add(ErrorIncorrectNumberOfBetasInReducedModel, "Incorrect number of betas in reduced linear regression model");
    add(ErrorMethodNotSupported, "Method not supported by the algorithm");
    add(ErrorIncorrectNumberOfFeatures, "Number of columns in numeric table is incorrect");
    add(ErrorIncorrectNumberOfObservations, "Number of rows in numeric table is incorrect");
    add(ErrorIncorrectSizeOfArray, "Incorrect size of array");
    add(ErrorNullParameterNotSupported, "Null parameter is not supported by the algorithm");
    add(ErrorIncorrectNumberOfArguments, "Number of arguments is incorrect");
    add(ErrorIncorrectInputNumericTable, "Input numeric table is incorrect");
    add(ErrorEmptyInputNumericTable, "Input numeric table is empty");
    add(ErrorIncorrectDataRange, "Data range is incorrect");
    add(ErrorPrecomputedStatisticsIndexOutOfRange, "Precomputed statistics index is out of range");
    add(ErrorIncorrectNumberOfInputNumericTables, "Incorrect number of input numeric tables");
    add(ErrorIncorrectNumberOfOutputNumericTables, "Incorrect number of output numeric tables");
    add(ErrorNullInputNumericTable, "Null input numeric table is not supported");
    add(ErrorNullOutputNumericTable, "Null output numeric table is not supported");
    add(ErrorNullModel, "Null model is not supported");
    add(ErrorInconsistentNumberOfRows, "Number of rows in provided numeric tables is inconsistent");
    add(ErrorInconsistentNumberOfColumns, "Number of columns in provided numeric tables is inconsistent");
    add(ErrorIncorrectSizeOfInputNumericTable, "Number of columns or rows in input numeric table is incorrect");
    add(ErrorIncorrectSizeOfOutputNumericTable, "Number of columns or rows in output numeric table is incorrect");
    add(ErrorIncorrectNumberOfRowsInInputNumericTable, "Number of rows in input numeric table is incorrect");
    add(ErrorIncorrectNumberOfColumnsInInputNumericTable, "Number of columns in input numeric table is incorrect");
    add(ErrorIncorrectNumberOfRowsInOutputNumericTable, "Number of rows in output numeric table is incorrect");
    add(ErrorIncorrectNumberOfColumnsInOutputNumericTable, "Number of columns in output numeric table is incorrect");
    add(ErrorIncorrectTypeOfInputNumericTable, "Incorrect type of input NumericTable");
    add(ErrorIncorrectTypeOfOutputNumericTable, "Incorrect type of output NumericTable");
    add(ErrorIncorrectNumberOfElementsInInputCollection, "Incorrect number of elements in input collection");
    add(ErrorIncorrectNumberOfElementsInResultCollection, "Incorrect number of elements in result collection");
    add(ErrorNullInput, "Input not set");
    add(ErrorNullResult, "Result not set");
    add(ErrorIncorrectParameter, "Incorrect parameter");
    add(ErrorModelNotFullInitialized, "Model is not full initialized");
    add(ErrorIncorrectIndex, "Index in collection is out of range");
    add(ErrorDataArchiveInternal, "Incorrect size of data block");
    add(ErrorNullPartialModel, "Null partial model is not supported");
    add(ErrorNullInputDataCollection, "Null input data collection is not supported");
    add(ErrorNullOutputDataCollection, "Null output data collection is not supported");
    add(ErrorNullPartialResult, "Partial result not set");
    add(ErrorNullLayerData, "Null layer data is not supported");
    add(ErrorIncorrectSizeOfLayerData, "Incorrect number of elements in layer data collection");
    add(ErrorNullAuxiliaryAlgorithm, "Null auxiliary algorithm");
    add(ErrorNullInitializationProcedure, "Null initialization procedure");
    add(ErrorIncorrectElementInCollection, "Incorrect element in collection");
    add(ErrorNullOptionalResult, "Null optional result");
    add(ErrorIncorrectOptionalResult, "Incorrect optional result");
    add(ErrorIncorrectOptionalInput, "Incorrect optional input");
    add(ErrorIncorrectNumberOfPartialClusters, "Incorrect number of partial clusters");
    add(ErrorIncorrectTotalNumberOfPartialClusters, "Incorrect total number of partial clusters");
    add(ErrorIncorrectDataCollectionSize, "Incorrect data collection size");
    add(ErrorIncorrectValueInTheNumericTable, "Incorrect value in the numeric table");
    add(ErrorIncorrectItemInDataCollection, "Incorrect item in data collection");
    add(ErrorNullPtr, "Null pointer in input arguments");
    add(ErrorUndefinedFeature, "Dictionary contains a undefined feature");
    add(ErrorCloneMethodFailed, "Cloning of algorithm failed");
    add(ErrorDataTypeNotSupported, "Data type not supported");
    add(ErrorNullByteInjection, "Null byte injection has been detected");
    add(ErrorBufferSizeIntegerOverflow, "Integer overflow has occurred");
    add(ErrorHyperparameterNotFound, "Cannot find a hyperparameter");
    add(ErrorHyperparameterBadValue, "Hyperparameter has bad value");

    // Environment errors: -2000..-2999
    add(ErrorCpuNotSupported, "CPU not supported");
    add(ErrorMemoryAllocationFailed, "Memory allocation failed");
    add(ErrorEmptyDataBlock, "Empty data block");
    add(ErrorMemoryCopyFailedInternal, "Memory copy internal error");

    // Workflow errors: -3000..-3999
    add(ErrorIncorrectCombinationOfComputationModeAndStep, "Incorrect combination of computation mode and computation step");
    add(ErrorDictionaryAlreadyAvailable, "Data Dictionary is already available");
    add(ErrorDictionaryNotAvailable, "Data Dictionary is not available");
    add(ErrorNumericTableNotAvailable, "Numeric Table is not available");
    add(ErrorNumericTableAlreadyAllocated, "Numeric Table was already allocated");
    add(ErrorNumericTableNotAllocated, "Numeric Table is not allocated");
    add(ErrorPrecomputedSumNotAvailable, "Precomputed sums are not available");
    add(ErrorPrecomputedMinNotAvailable, "Precomputed minimum values are not available");
    add(ErrorPrecomputedMaxNotAvailable, "Precomputed maximum values are not available");
    add(ErrorServiceMicroTableInternal, "Numeric Table internal error");
    add(ErrorEmptyCSRNumericTable, "CSR Numeric Table is empty");
    add(ErrorEmptyHomogenNumericTable, "Homogeneous Numeric Table is empty");
    add(ErrorSourceDataNotAvailable, "Source data is not available");
    add(ErrorEmptyDataSource, "Data source is empty");
    add(ErrorIncorrectClassLabels, "Class labels provided to the algorithm are incorrect");
    add(ErrorIncorrectSizeOfModel, "Incorrect size of model");
    add(ErrorIncorrectTypeOfModel, "Incorrect type of model");
    add(ErrorIncorrectErrorcodeFromGenerator, "Incorrect error code is returned from data generator");
    add(ErrorLeapfrogUnsupported, "Leapfrog method is not supported by generator");
    add(ErrorSkipAheadUnsupported, "SkipAhead method is not supported by generator");
    add(ErrorFeatureNamesNotAvailable, "Feature names are not available for feature modifier");
    add(ErrorEngineNotSupported, "Engine not supported");

    // Common computation errors: -4000...
    add(ErrorInputSigmaMatrixHasNonPositiveMinor, "Input sigma matrix has non positive minor");
    add(ErrorInputSigmaMatrixHasIllegalValue, "Input sigma matrix has illegal value");
    add(ErrorIncorrectInternalFunctionParameter, "Incorrect parameter in internal function call");
    add(ErrorUserCancelled, "Computation cancelled at user's request");

    /* Apriori algorithm errors -5000..-5199 */
    add(ErrorAprioriIncorrectItemsetTableSize, "Number of rows in the output table containing 'large' item sets is too small");
    add(ErrorAprioriIncorrectSupportTableSize, "Number of rows in the output table containing 'large' item sets support values is too small");
    add(ErrorAprioriIncorrectLeftRuleTableSize, "Number of rows in the output table containing left parts of the association rules is too small");
    add(ErrorAprioriIncorrectRightRuleTableSize, "Number of rows in the output table containing right parts of the association rules is too small");
    add(ErrorAprioriIncorrectConfidenceTableSize, "Number of rows in the output table containing association rules confidence is too small");
    add(ErrorAprioriIncorrectInputData, "Incorrect input data");

    // BrownBoost errors: -5200..-5399
    add(ErrorInconsistentNumberOfClasses, "Inconsistent number of classes between the boosting algorithm and the weak learner");

    // Cholesky errors: -5400..-5599
    add(ErrorCholeskyInternal, "Cholesky internal error");
    add(ErrorInputMatrixHasNonPositiveMinor, "Input matrix has non positive minor");

    // Covariance errors: -5600..-5799
    add(ErrorCovarianceInternal, "Covariance internal error");

    // Distance errors: -5800..-5999

    // EM errors: -6000..-6099
    add(ErrorEMMatrixInverse, "Sigma matrix on M-step cannot be inverted");
    add(ErrorEMIncorrectToleranceToConverge, "Incorrect value of tolerance to converge in EM parameter");
    add(ErrorEMIllConditionedCovarianceMatrix, "Ill-conditioned covariance matrix");
    add(ErrorEMIncorrectMaxNumberOfIterations, "Incorrect maximum number of iterations value in EM parameter");
    add(ErrorEMNegativeDefinedCovarianceMartix, "Negative-defined covariance matrix");
    add(ErrorEMEmptyComponent, "Empty component during computation");
    add(ErrorEMCovariance, "Error during covariance computation for component on M step");
    add(ErrorEMIncorrectNumberOfComponents, "Incorrect number of components value in EM parameter");

    // EM initialization errors: -6100..-6199
    add(ErrorEMInitNoTrialConverges, "No trial of internal EM start converges");
    add(ErrorEMInitIncorrectToleranceToConverge, "Incorrect tolerance to converge value in EM initialization parameter");
    add(ErrorEMInitIncorrectDepthNumberIterations, "Incorrect depth number of iterations value in EM init parameter");
    add(ErrorEMInitIncorrectNumberOfTrials, "Incorrect number of trials value in EM initialization parameter");
    add(ErrorEMInitIncorrectNumberOfComponents, "Incorrect number of components value in EM initialization parameter");
    add(ErrorEMInitInconsistentNumberOfComponents,
        "Inconsistent number of component: number of observations should be greater than number of components");
    add(ErrorVarianceComputation, "Error during variance computation");

    // KernelFunction errors: -6200..-6399

    // KMeans errors: -6400..-6599
    add(ErrorKMeansNumberOfClustersIsTooLarge, "Number of clusters exceeds the number of points");

    // Linear Rergession errors: -6600..-6799
    add(ErrorLinearRegressionInternal, "Linear Regression internal error");
    add(ErrorNormEqSystemSolutionFailed, "Failed to solve the system of normal equations");
    add(ErrorLinRegXtXInvFailed, "Failed to invert Xt*X matrix");

    // LogitBoots errors: -6800..-6999

    // LowOrderMoments errors: -7000..-7199
    add(ErrorLowOrderMomentsInternal, "Low Order Moments internal error");

    // MultiClassClassifier errors: -7200..-7399
    add(ErrorIncorrectNumberOfClasses, "Number of classes provided to multi-class classifier is too small");
    add(ErrorMultiClassNullTwoClassTraining, "Null two-class classifier training algorithm is not supported");
    add(ErrorMultiClassFailedToTrainTwoClassClassifier, "Failed to train a model of two-class classifier");
    add(ErrorMultiClassFailedToComputeTwoClassPrediction, "Failed to compute prediction based on two-class classifier model");

    // NaiveBayes errors: -7400..-7599
    add(ErrorEmptyInputCollection, "Naive Bayes: Input collection is empty");
    add(ErrorNaiveBayesIncorrectModel, "Naive Bayes: Input model is not consistent with the number of classes");

    // OutlierDetection errors: -7600..-7799
    add(ErrorOutlierDetectionInternal, "Outlier Detection internal error");

    /* PCA errors: -7800..-7999 */
    add(ErrorPCAFailedToComputeCorrelationEigenvalues, "Failed to compute eigenvalues of the correlation matrix");
    add(ErrorPCACorrelationInputDataTypeSupportsOfflineModeOnly, "This type of the input data supports only offline mode of the computations");
    add(ErrorIncorrectCrossProductTableSize, "Number of columns or rows in cross-product numeric table is incorrect");
    add(ErrorCrossProductTableIsNotSquare, "Number of columns or rows in cross-product numeric table is not equal");
    add(ErrorInputCorrelationNotSupportedInOnlineAndDistributed,
        "Input correlation matrix is not supported in online and distributed computation modes");
    add(ErrorIncorrectNComponents, "Incorrect nComponents parameter: nComponents should be less or equal to number of columns in testing dataset");

    // QR errors: -8000..-8199
    add(ErrorQRInternal, "QR internal error");

    // Stump errors: -8200..-8399
    add(ErrorStumpIncorrectSplitFeature, "Incorrect split feature: split feature should be less than number of columns in testing dataset");
    add(ErrorStumpInvalidInputCategoricalData,
        "Invalid stump training data: all features in the input table are categorical and each feature has < 2 categories");

    // LCN errors: -8400..-8599
    add(ErrorLCNinnerConvolution, "Error in convolution 2d layer");

    // SVM errors: -8600..-8799
    add(ErrorSVMPredictKernerFunctionCall, "SVM predict: error in kernel function call. Details are as follows.");

    // WeakLearner errors: -8800..-8999
    add(ErrorIncorrectWeakLearnerClassificationAlgorithm, "Weak learner can not be casted to classifier algorithm");
    add(ErrorIncorrectWeakLearnerRegressionAlgorithm, "Weak learner can not be casted to regression algorithm");
    add(ErrorIncorrectWeakLearnerClassificationModel, "Weak learner's model can not be casted to classifier model");
    add(ErrorIncorrectWeakLearnerRegressionModel, "Weak learner's model can not be casted to regression model");

    // Compression errors: -9000..-9199
    add(ErrorCompressionNullInputStream, "Null input stream is not supported");
    add(ErrorCompressionNullOutputStream, "Null output stream is not supported");

    add(ErrorCompressionEmptyInputStream, "Input stream of size 0 is not supported");
    add(ErrorCompressionEmptyOutputStream, "Output stream of size 0 is not supported");

    add(ErrorZlibInternal, "Zlib internal error");
    add(ErrorZlibDataFormat, "Input compressed stream is in wrong format, corrupted or contains not a whole number of compressed blocks");
    add(ErrorZlibParameters, "Unsupported Zlib parameters");
    add(ErrorZlibMemoryAllocationFailed, "Internal Zlib memory allocation failed");
    add(ErrorZlibNeedDictionary, "Specific dictionary is needed for decompression, currently unsupported Zlib feature");

    add(ErrorBzip2Internal, "Bzip2 internal error");
    add(ErrorBzip2DataFormat, "Input compressed stream is in wrong format, corrupted or contains not a whole number of compressed blocks");
    add(ErrorBzip2Parameters, "Unsupported Bzip2 parameters");
    add(ErrorBzip2MemoryAllocationFailed, "Internal Bzip2 memory allocation failed");

    add(ErrorLzoInternal, "LZO internal error");
    add(ErrorLzoOutputStreamSizeIsNotEnough, "Size of output stream is not enough to start compression");
    add(ErrorLzoDataFormat, "Input compressed stream is in wrong format or corrupted");
    add(ErrorLzoDataFormatLessThenHeader, "Size of input compressed stream is less then compressed block header size");
    add(ErrorLzoDataFormatNotFullBlock, "Input compressed stream contains not a whole number of compressed blocks");

    add(ErrorRleInternal, "RLE internal error");
    add(ErrorRleOutputStreamSizeIsNotEnough, "Size of output stream is not enough to start compression");
    add(ErrorRleDataFormat, "Input compressed stream is in wrong format or corrupted");
    add(ErrorRleDataFormatLessThenHeader, "Size of input compressed stream is less then compressed block header size");
    add(ErrorRleDataFormatNotFullBlock, "Input compressed stream contains not a whole number of compressed blocks");

    // Min-max normalization errors: -9400..-9499
    add(ErrorLowerBoundGreaterThanOrEqualToUpperBound, "Lower bound parameter greater than or equal to upper bound");

    // Quantile error: -10000..-11000
    add(ErrorQuantileOrderValueIsInvalid, "Quantile order value is invalid");
    add(ErrorQuantilesInternal, "Quantile internal error");

    // ALS errors: -11000..-12000
    add(ErrorALSInternal, "ALS algorithm failed to solve a system of normal equations");
    add(ErrorALSInconsistentSparseDataBlocks, "Failed to find a non-zero value with needed indices in a sparse data block");

    // Sorting error: -12000..-13000
    add(ErrorSorting, "Cannot sort the numeric table");

    // SGD error: -13000..-14000
    add(ErrorNegativeLearningRate, "Negative learning rate");

    // Normalization errors: -14000..-15000
    add(ErrorMeanAndStandardDeviationComputing, "Computation of mean and standard deviation failed");
    add(ErrorNullVariance, "Failed to normalize data in column: it has null variance");
    add(ErrorMinAndMaxComputing, "Failed to compute minimum and maximum of each column");

    // Sum of functions error: -14000..-15000
    add(ErrorZeroNumberOfTerms, "Number of terms can not be zero");

    // Convolution2d error: -16000..-17000
    add(ErrorConvolutionInternal, "Convolution layer internal error");
    add(ErrorIncorrectKernelSise1, "Convolution2d backward: incorrect parameter kernelSize1");
    add(ErrorIncorrectKernelSise2, "Convolution2d backward: incorrect parameter kernelSize2");

    // Ridge Regression errors: -17000..-17999
    add(ErrorRidgeRegressionInternal, "Ridge Regression internal error");
    add(ErrorRidgeRegressionNormEqSystemSolutionFailed, "Failed to solve the system of normal equations");
    add(ErrorRidgeRegressionInvertFailed, "Failed to invert matrix");

    // Pivoted QR errors: -19000..-19199
    add(ErrorPivotedQRInternal, "Pivoted QR internal error");

    // Decision forest error: -20000..-20099
    add(ErrorDFBootstrapVarImportanceIncompatible, "Parameter 'bootstrap' is incompatible with requested variable importance type");
    add(ErrorDFBootstrapOOBIncompatible, "Parameter 'bootstrap' is incompatible with requested OOB result (no out-of-bag observations)");

    // K-Nearest Neighbors errors: -21000..21999
    add(ErrorKNNInternal, "K-Nearest Neighbors internal error");

    // GBT error: -30000..-30099
    add(ErrorGbtIncorrectNumberOfTrees, "Number of trees in the model is not consistent with the number of classes");
    add(ErrorGbtPredictIncorrectNumberOfIterations, "Number of iterations value in GBT parameter is not consistent with the model");
    add(ErrorGbtPredictShapOptions, "Incompatible SHAP options. Can calculate either contributions or interactions, not both");
    add(ErrorGbtPredictShapMulticlassNotSupported, "Multiclass classification SHAP values not supported.");

    //Math errors: -90000..-90099
    add(ErrorDataSourseNotAvailable, "ErrorDataSourseNotAvailable");
    add(ErrorHandlesSQL, "ErrorHandlesSQL");
    add(ErrorODBC, "ErrorODBC");
    add(ErrorSQLstmtHandle, "ErrorSQLstmtHandle");
    add(ErrorOnFileOpen, "Error on file open");
    add(ErrorOnFileRead, "Error on file read");

    add(ErrorKDBNoConnection, "ErrorKDBNoConnection");
    add(ErrorKDBWrongCredentials, "ErrorKDBWrongCredentials");
    add(ErrorKDBNetworkError, "ErrorKDBNetworkError");
    add(ErrorKDBServerError, "ErrorKDBServerError");
    add(ErrorKDBTypeUnsupported, "ErrorKDBTypeUnsupported");
    add(ErrorKDBWrongTypeOfOutput, "ErrorKDBWrongTypeOfOutput");

    //Distributions errors -90100..90200
    add(ErrorIncorrectEngineParameter, "Incorrect engine parameter in distribution");

    // Quality metrics errors -90201..-90301
    add(ErrorEmptyInputAlgorithmsCollection, "Quality metric set: Input algorithms collection is empty");

    // Group of SYCL-related errors -90900..-90999
    add(ErrorEmptyBuffer, "Cannot access empty buffer");
    add(ErrorAccessUSMPointerOnOtherDevice, "Cannot access USM pointer from the other device");

    // Other errors: -100000..
    add(ErrorObjectDoesNotSupportSerialization, "SerializationIface is not implemented or implemented incorrectly");
    add(ErrorExecutionContext, "Execution context error");
    add(ErrorHashTableCollision, "Hash table collision happened. Please, increase the buffer size.");

    add(ErrorCouldntAttachCurrentThreadToJavaVM, "Couldn't attach current thread to Java VM");
    add(ErrorCouldntCreateGlobalReferenceToJavaObject, "Couldn't create global reference to Java object");
    add(ErrorCouldntFindJavaMethod, "Couldn't find Java method");
    add(ErrorCouldntFindClassForJavaObject, "Couldn't find class for Java object");

    add(UnknownError, "UnknownError");

    add(NoErrorMessageFound, "NoErrorMessageFound");
    add(ErrorMethodNotImplemented, "Method is not implemented");

    add(ErrorDeviceSupportNotImplemented, "Device support for the algorithm isn't implemented");

    add(ErrorIncorrectOffset, "Incorrect offset");
    add(ErrorIterativeSolverIncorrectMaxNumberOfIterations, "Incorrect maximum number of iterations value in solver");
    add(ErrorIncorrectNumberOfTerms, "Incorrect number of summands (terms) in objective function");
    add(ErrorIncorrectNumberOfNodes, "Incorrect number of nodes");
    add(ErrorInconsistenceModelAndBatchSizeInParameter, "Inconsistence of model and batch size parameter in optimization solver");

    add(ErrorCanNotLoadDynamicLibrary, "Failure during loading of dynamic library");
    add(ErrorCanNotLoadDynamicLibrarySymbol, "Failure during loading symbol from dynamic library");
}

void ErrorDetailCollection::parseResourceFile()
{
    add(NoErrorMessageDetailFound, "NoErrorMessageDetailFound");
    add(Row, "Row");
    add(Column, "Column");
    add(Rank, "Rank");
    add(StatisticsName, "Statistics name");
    add(Method, "Method");
    add(Iteration, "Iteration");
    add(Component, "Component");
    add(Minor, "Matrix minor");
    add(ArgumentName, "Argument name");
    add(ElementInCollection, "ElementInCollection");
    add(ParameterName, "Parameter name");
    add(OptionalInput, "Optional input");
    add(OptionalResult, "Optional result");
    add(ExpectedValue, "Expected");
    add(ActualValue, "Actual");
    add(Sycl, "Sycl error");
    add(OpenCL, "OpenCL error");
    add(Key, "Key");
}

} // namespace services

/////////////////////////////////////////////////////////////////////////////////////////
// Mutex
/////////////////////////////////////////////////////////////////////////////////////////
Mutex::Mutex() : _impl(nullptr)
{
    _impl = _daal_new_mutex();
}

Mutex::~Mutex()
{
    if (_impl) _daal_del_mutex(_impl);
}

void Mutex::lock()
{
    if (_impl) _daal_lock_mutex(_impl);
}

void Mutex::unlock()
{
    if (_impl) _daal_unlock_mutex(_impl);
}

/////////////////////////////////////////////////////////////////////////////////////////
// SafeStatus
/////////////////////////////////////////////////////////////////////////////////////////
SafeStatus::SafeStatus() {}

SafeStatus::SafeStatus(const services::Status & s) : _val(s) {}

bool SafeStatus::ok() const
{
    AUTOLOCK(_m);
    return _val.ok();
}

SafeStatus & SafeStatus::add(services::ErrorID id)
{
    {
        AUTOLOCK(_m);
        _val.add(id);
    }
    return *this;
}

SafeStatus & SafeStatus::add(const services::ErrorPtr & e)
{
    {
        AUTOLOCK(_m);
        _val.add(e);
    }
    return *this;
}

SafeStatus & SafeStatus::add(const services::Status & s)
{
    if (!s)
    {
        AUTOLOCK(_m);
        _val.add(s);
    }
    return *this;
}

services::Status SafeStatus::detach()
{
    AUTOLOCK(_m);
    services::Status s = _val;
    _val               = services::Status();
    return s;
}

} // namespace daal
