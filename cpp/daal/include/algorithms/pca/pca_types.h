/* file: pca_types.h */
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
//  Implementation of PCA algorithm interface.
//--
*/

#ifndef __PCA_TYPES_H__
#define __PCA_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/covariance/covariance_batch.h"
#include "algorithms/covariance/covariance_online.h"
#include "algorithms/covariance/covariance_distributed.h"
#include "algorithms/normalization/zscore.h"

namespace daal
{
namespace algorithms
{
/**
* @defgroup pca Principal Component Analysis
* \copydoc daal::algorithms::pca
* @ingroup analysis
* @{
*/
/**
* \brief Contains classes for computing the results of the principal component analysis (PCA) algorithm
*/
namespace pca
{
/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__METHOD"></a>
    * Available methods for computing the PCA algorithm
    */
enum Method
{
    correlationDense = 0, /*!< PCA Correlation method */
    defaultDense     = 0, /*!< PCA Default method */
    svdDense         = 1  /*!< PCA SVD method */
};

/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__INPUTDATASETID"></a>
    * Available identifiers of input dataset objects for the PCA algorithm
    */
enum InputDatasetId
{
    data, /*!< Input data table */
    lastInputDatasetId = data
};

/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__INPUTCORRELATIONID"></a>
    * Available identifiers of input objects for the PCA Correlation algorithm
    */
enum InputCorrelationId
{
    correlation, /*!< Input correlation table */
    lastInputCorrelationId = correlation
};

/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__STEP2MASTERINPUTID"></a>
    * Available identifiers of input objects for the PCA algorithm on the second step in the distributed processing mode
    */
enum Step2MasterInputId
{
    partialResults, /*!< Collection of partial results computed on local nodes */
    lastStep2MasterInputId = partialResults
};

/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__PARTIALCORRELATIONRESULTID"></a>
    * Available identifiers of partial results of the PCA Correlation algorithm
    */
enum PartialCorrelationResultId
{
    nObservationsCorrelation, /* Number of processed observations */
    crossProductCorrelation,  /* Cross-product of the processed data */
    sumCorrelation,           /* Feature sums of the processed data */
    lastPartialCorrelationResultId = sumCorrelation
};

/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__PARTIALSVDTABLERESULTID"></a>
    * Available identifiers of partial results of the PCA SVD algorithm
    */
enum PartialSVDTableResultId
{
    nObservationsSVD, /* Number of processed observations */
    sumSVD,           /* Feature sums of the processed data */
    sumSquaresSVD,    /* Feature sums of squares of the processed data */
    lastPartialSVDTableResultId = sumSquaresSVD
};

/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__PARTIALSVDCOLLECTIONRESULTID"></a>
    * Available identifiers of partial results of the PCA SVD  algorithm
    */
enum PartialSVDCollectionResultId
{
    auxiliaryData = lastPartialSVDTableResultId + 1, /*!< Auxiliary data of the PCA SVD method */
    distributedInputs, /*!< Auxiliary data of the PCA SVD method on the second step in the distributed processing mode */
    lastPartialSVDCollectionResultId = distributedInputs
};

/**
    * <a name="DAAL-ENUM-ALGORITHMS__PCA__RESULTID"></a>
    * Available identifiers of the results of the PCA algorithm
    */
enum ResultId
{
    eigenvalues,  /*!< Eigenvalues of the correlation matrix */
    eigenvectors, /*!< Eigenvectors of the correlation matrix */
    means,        /*!< Mean values */
    variances,    /*!< Variances */
    lastResultId = variances
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__PCA__RESULTCOLLECTIONID"></a>
* Available identifiers of the result collections of the PCA algorithm
*/
enum ResultCollectionId
{
    dataForTransform /*!< Eigenvalues, means and variances */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__PCA__RESULTOCOMPUTETID"></a>
* Available identifiers of optional results of the PCA batch algorithms
* @ingroup zscore
*/
enum ResultToComputeId
{
    none       = 0ULL,
    mean       = 0x00000001ULL, /*!< Numeric table of size 1 x p with the mean values of features >*/
    variance   = 0x00000002ULL, /*!< Numeric table of size 1 x p with the variances of features >*/
    eigenvalue = 0x00000004ULL  /*!< Numeric table of size 1 x p with the always computed eigenvalues>*/
};

/**
    * \brief Contains version 1.0 of Intel(R) oneAPI Data Analytics Library interface.
    */
namespace interface1
{
/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__INPUTIFACE"></a>
    * \brief Abstract class that specifies interface for classes that declare input of the PCA algorithm */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements);
    InputIface(const InputIface & other);

    /**
        * Returns the number of columns in the input data set
        * \return Number of columns in the input data set
        */
    virtual size_t getNFeatures() const = 0;

    /**
    * Returns flag defining whether the input data contains correlation matrix or not
    * \return Flag defining whether the input data contains correlation matrix or not
    */
    virtual bool isCorrelation() const { return _isCorrelation; };

    virtual ~InputIface() {};

protected:
    bool _isCorrelation;
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__INPUT"></a>
    * \brief Input objects for the PCA algorithm
    */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);

    virtual ~Input() {};

    /**
    * Returns the input object of the PCA algorithm
        * \param[in] id    Identifier of the input object
        * \return          Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputDatasetId id) const;

    /**
        * Sets input dataset for the PCA algorithm
        * \param[in] id      Identifier of the input object
        * \param[in] value   Pointer to the input object
        */
    void set(InputDatasetId id, const data_management::NumericTablePtr & value);

    /**
        * Sets input correlation matrix for the PCA algorithm
        * \param[in] id      Identifier of the input object
        * \param[in] value   Pointer to the input object
        */
    void set(InputCorrelationId id, const data_management::NumericTablePtr & value);

    /**
        * Returns the number of columns in the input data set
        * \return Number of columns in the input data set
        */
    size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks input algorithm parameters
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computation method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALRESULTBASE"></a>
    * \brief Provides interface to access partial results obtained with the compute() method of the
    *        PCA algorithm in the online or distributed processing mode
    */
class PartialResultBase : public daal::algorithms::PartialResult
{
public:
    PartialResultBase(const size_t nElements) : daal::algorithms::PartialResult(nElements) {};

    virtual size_t getNFeatures() const = 0;

    virtual ~PartialResultBase() {};
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__PARTIALRESULT"></a>
    * \brief Provides methods to access partial results obtained with the compute() method of the
    *        PCA algorithm in the online or distributed processing mode
    */
template <Method method>
class PartialResult : public PartialResultBase
{};

/**
    * <a name="DAAL-CLASS-PCA__PARTIALRESULT"></a>
    * \brief Provides methods to access partial results obtained with the compute() method of the PCA Correlation algorithm
    *        in the online or distributed processing mode
    */
template <>
class DAAL_EXPORT PartialResult<daal::algorithms::pca::correlationDense> : public PartialResultBase
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult<daal::algorithms::pca::correlationDense>)
    PartialResult();

    /**
        * Gets partial results of the PCA Correlation algorithm
        * \param[in] id    Identifier of the input object
        * \return          Input object that corresponds to the given identifier
        */
    data_management::NumericTablePtr get(PartialCorrelationResultId id) const;

    virtual size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
        * Sets partial result of the PCA Correlation algorithm
        * \param[in] id      Identifier of the result
        * \param[in] value   Pointer to the object
        */
    void set(const PartialCorrelationResultId id, const data_management::NumericTablePtr & value);

    virtual ~PartialResult() {};

    /**
    * Checks partial results of the PCA Correlation algorithm
    * \param[in] input      %Input object of the algorithm
    * \param[in] parameter  Algorithm %parameter
    * \param[in] method     Computation method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks partial results of the PCA Ccorrelation algorithm
    * \param[in] par        Algorithm %parameter
    * \param[in] method     Computation method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
        * Allocates memory to store partial results of the PCA  SVD algorithm
        * \param[in] input     Pointer to an object containing input data
        * \param[in] parameter Pointer to the structure of algorithm parameters
        * \param[in] method    Computation method
        * \return Status of allocation
        */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
        * Initializes memory to store partial results of the PCA  SVD algorithm
        * \param[in] input     Pointer to an object containing input data
        * \param[in] parameter Pointer to the structure of algorithm parameters
        * \param[in] method    Computation method
        * \return Status of initialization
        */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

protected:
    services::Status checkImpl(size_t nFeatures) const;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
    * <a name="DAAL-CLASS-PCA__PARTIALRESULT"></a>
    * \brief Provides methods to access partial results obtained with the compute() method of PCA SVD algorithm
    *         in the online or distributed processing mode
    */
template <>
class DAAL_EXPORT PartialResult<daal::algorithms::pca::svdDense> : public PartialResultBase
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult<daal::algorithms::pca::svdDense>)
    PartialResult();

    /**
    * Gets partial results of the PCA SVD algorithm
        * \param[in] id    Identifier of the input object
        * \return          Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(PartialSVDTableResultId id) const;

    virtual size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Gets partial results of the PCA SVD algorithm
        * \param[in] id    Identifier of the input object
        * \return          Input object that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(PartialSVDCollectionResultId id) const;

    /**
    * Gets partial results of the PCA SVD algorithm
        * \param[in] id            Identifier of the input object
        * \param[in] elementId     Identifier of the collection element
        * \return                  Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(PartialSVDCollectionResultId id, const size_t & elementId) const;

    /**
        * Sets partial result of the PCA SVD algorithm
        * \param[in] id      Identifier of the result
        * \param[in] value   Pointer to  the object
        */
    void set(PartialSVDTableResultId id, const data_management::NumericTablePtr & value);

    /**
        * Sets partial result of the PCA SVD algorithm
        * \param[in] id      Identifier of the result
        * \param[in] value   Pointer to the object
        */
    void set(PartialSVDCollectionResultId id, const data_management::DataCollectionPtr & value);

    /**
        * Adds partial result of the PCA SVD algorithm
        * \param[in] id      Identifier of the argument
        * \param[in] value   Pointer to the object
        */
    void add(const PartialSVDCollectionResultId & id, const data_management::DataCollectionPtr & value);

    /**
    * Checks partial results of the PCA SVD algorithm
    * \param[in] input      %Input of algorithm
    * \param[in] parameter  %Parameter of algorithm
    * \param[in] method     Computation method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks partial results of the PCA SVD algorithm
    * \param[in] method     Computation method
    * \param[in] par        %Parameter of algorithm
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    virtual ~PartialResult() {};

    /**
        * Allocates memory to store partial results of the PCA  SVD algorithm
        * \param[in] input     Pointer to an object containing input data
        * \param[in] parameter Pointer to the structure of algorithm parameters
        * \param[in] method    Computation method
        * \return Status of allocation
        */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
        * Initializes memory to store partial results of the PCA  SVD algorithm
        * \param[in] input     Pointer to an object containing input data
        * \param[in] parameter Pointer to the structure of algorithm parameters
        * \param[in] method    Computation method
        * \return Status of initialization
        */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

protected:
    services::Status checkImpl(size_t nFeatures) const;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__BASEPARAMETER"></a>
    * \brief Class that specifies the common parameters of the PCA algorithm
    */
template <typename algorithmFPType, Method method = correlationDense>
class DAAL_EXPORT BaseParameter : public daal::algorithms::Parameter
{
public:
    /** Constructs PCA parameters */
    BaseParameter();
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER"></a>
    * \brief Class that specifies the parameters of the PCA algorithm in the online computing mode
    */
template <typename algorithmFPType, Method method>
class DAAL_EXPORT OnlineParameter : public BaseParameter<algorithmFPType, method>
{};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
    * \brief Class that specifies the parameters of the PCA Correlation algorithm in the online computing mode
    */
template <typename algorithmFPType>
class OnlineParameter<algorithmFPType, correlationDense> : public BaseParameter<algorithmFPType, correlationDense>
{
public:
    /** Constructs PCA parameters */
    OnlineParameter(const services::SharedPtr<covariance::OnlineImpl> & covarianceForOnlineParameter =
                        services::SharedPtr<covariance::Online<algorithmFPType, covariance::defaultDense> >(
                            new covariance::Online<algorithmFPType, covariance::defaultDense>()));

    services::SharedPtr<covariance::OnlineImpl> covariance; /*!< Pointer to Online covariance */

    /**
    * Checks online parameter of the PCA correlation algorithm
    * \return Errors detected while checking
    */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__ONLINEPARAMETER_ALGORITHMFPTYPE_SVDDENSE"></a>
    * \brief Class that specifies the parameters of the PCA SVD algorithm in the online computing mode
    */
template <typename algorithmFPType>
class OnlineParameter<algorithmFPType, svdDense> : public BaseParameter<algorithmFPType, svdDense>
{
public:
    /** Constructs PCA parameters */
    OnlineParameter();

    /**
    * Checks online parameter of the PCA SVD algorithm
    * \return Errors detected while checking
    */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDPARAMETER"></a>
    * \brief Class that specifies the parameters of the PCA algorithm in the distributed computing mode
    */
template <ComputeStep step, typename algorithmFPType, Method method>
class DistributedParameter : public BaseParameter<algorithmFPType, method>
{};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDPARAMETER_STEP2MASTER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
    * \brief Class that specifies the parameters of the PCA Correlation algorithm in the distributed computing mode
    */
template <typename algorithmFPType>
class DistributedParameter<step2Master, algorithmFPType, correlationDense> : public BaseParameter<algorithmFPType, correlationDense>
{
public:
    /** Constructs PCA parameters */
    DistributedParameter(const services::SharedPtr<covariance::DistributedIface<step2Master> > & covarianceForDistributedParameter =
                             services::SharedPtr<covariance::Distributed<step2Master, algorithmFPType, covariance::defaultDense> >(
                                 new covariance::Distributed<step2Master, algorithmFPType, covariance::defaultDense>()));

    services::SharedPtr<covariance::DistributedIface<step2Master> > covariance; /*!< Pointer to Distributed covariance */

    /**
    * Checks distributed parameter of the PCA correlation algorithm
    * \return Errors detected while checking
    */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTEDINPUT"></a>
    * \brief Input objects for the PCA algorithm in the distributed processing mode
    */
template <Method method>
class DistributedInput
{};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED_INPUT"></a>
    * \brief Input objects for the PCA Correlation algorithm in the distributed processing mode
    */
template <>
class DistributedInput<correlationDense> : public InputIface
{
public:
    DAAL_EXPORT DistributedInput();
    DAAL_EXPORT DistributedInput(const DistributedInput & other);
    DAAL_EXPORT DistributedInput operator=(const DistributedInput & other);

    /**
        * Sets input objects for the PCA on the second step in the distributed processing mode
        * \param[in] id    Identifier of the input object
        * \param[in] ptr   Input object that corresponds to the given identifier
        */
    DAAL_EXPORT void set(Step2MasterInputId id, const data_management::DataCollectionPtr & ptr);

    /**
        * Gets input objects for the PCA on the second step in the distributed processing mode
        * \param[in] id    Identifier of the input object
        * \return          Input object that corresponds to the given identifier
        */
    data_management::DataCollectionPtr get(Step2MasterInputId id) const;

    /**
        * Retrieves specific partial result from the input objects of the PCA algorithm on the second step in the distributed processing mode
        * \param[in] id      Identifier of the partial result
        */
    services::SharedPtr<PartialResult<correlationDense> > getPartialResult(size_t id) const;

    /**
        * Adds input objects of the PCA algorithm on the second step in the distributed processing mode
        * \param[in] id      Identifier of the argument
        * \param[in] value   Pointer to the argument
        */
    DAAL_EXPORT void add(Step2MasterInputId id, const services::SharedPtr<PartialResult<correlationDense> > & value);

    /**
        * Returns the number of columns in the input data set
        * \return Number of columns in the input data set
        */
    size_t getNFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks the input of the PCA algorithm
    * \param[in] parameter Algorithm %parameter
    * \param[in] method    Computation  method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__DISTRIBUTED_INPUT"></a>
    * \brief Input objects of the PCA SVD algorithm in the distributed processing mode
    */
template <>
class DistributedInput<svdDense> : public InputIface
{
public:
    DAAL_EXPORT DistributedInput();
    DAAL_EXPORT DistributedInput(const DistributedInput & other);

    /**
        * Sets input objects for the PCA on the second step in the distributed processing mode
        * \param[in] id    Identifier of the input object
        * \param[in] ptr   Input object that corresponds to the given identifier
        */
    DAAL_EXPORT void set(Step2MasterInputId id, const data_management::DataCollectionPtr & ptr);

    /**
        * Gets input objects for the PCA algorithm on the second step in the distributed processing mode
        * \param[in] id    Identifier of the input object
        * \return          Input object that corresponds to the given identifier
        */
    data_management::DataCollectionPtr get(Step2MasterInputId id) const;

    /**
        * Adds input objects of the PCA algorithm on the second step in the distributed processing mode
        * \param[in] id      Identifier of the input object
        * \param[in] value   Pointer to the input object
        */
    DAAL_EXPORT void add(Step2MasterInputId id, const services::SharedPtr<PartialResult<svdDense> > & value);

    /**
        * Retrieves specific partial result from the input objects of the PCA algorithm on the second step in the distributed processing mode
        * \param[in] id      Identifier of the partial result
        */
    services::SharedPtr<PartialResult<svdDense> > getPartialResult(size_t id) const;

    /**
    * Checks the input of the PCA algorithm
    * \param[in] parameter Algorithm %parameter
    * \param[in] method    Computation  method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

    /**
        * Returns the number of columns in the input data set
        * \return Number of columns in the input data set
        */
    size_t getNFeatures() const DAAL_C11_OVERRIDE;
};

} // namespace interface1

/**
    * \brief Contains version 3.0 of Intel(R) oneAPI Data Analytics Library interface.
    */
namespace interface3
{
/**
* <a name="DAAL-CLASS-ALGORITHMS__PCA__BASEBATCHPARAMETER"></a>
* \brief Class that specifies the common parameters of the PCA Batch algorithms
*/
class DAAL_EXPORT BaseBatchParameter : public daal::algorithms::Parameter
{
public:
    /** Constructs PCA parameters */
    BaseBatchParameter();

    DAAL_UINT64 resultsToCompute; /*!< 64 bit integer flag that indicates the results to compute */
    size_t nComponents;           /*!< number of components for reduced implementation (applicable for batch mode only) */
    bool isDeterministic;         /*!< sign flip if required */
    bool doScale;                 /*!< scaling if required */
    bool isCorrelation;           /*!< correlation is provided */
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER"></a>
* \brief Class that specifies the parameters of the PCA algorithm in the batch computing mode
*/
template <typename algorithmFPType, Method method>
class DAAL_EXPORT BatchParameter
{};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER_ALGORITHMFPTYPE_CORRELATIONDENSE"></a>
    * \brief Class that specifies the parameters of the PCA Correlation algorithm in the batch computing mode
    */
template <typename algorithmFPType>
class BatchParameter<algorithmFPType, correlationDense> : public BaseBatchParameter
{
public:
    /** Constructs PCA parameters */
    BatchParameter(const services::SharedPtr<covariance::BatchImpl> & covarianceForBatchParameter =
                       services::SharedPtr<covariance::Batch<algorithmFPType, covariance::defaultDense> >(
                           new covariance::Batch<algorithmFPType, covariance::defaultDense>()));

    services::SharedPtr<covariance::BatchImpl> covariance; /*!< Pointer to batch covariance */

    /**
    * Checks batch parameter of the PCA correlation algorithm
    * \return Errors detected while checking
    */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__PCA__BATCHPARAMETER_ALGORITHMFPTYPE_SVDDENSE"></a>
* \brief Class that specifies the parameters of the PCA SVD algorithm in the batch computing mode
*/
template <typename algorithmFPType>
class BatchParameter<algorithmFPType, svdDense> : public BaseBatchParameter
{
public:
    /** Constructs PCA parameters */
    BatchParameter(const services::SharedPtr<normalization::zscore::BatchImpl> & normalizationForBatchParameter =
                       services::SharedPtr<normalization::zscore::Batch<algorithmFPType, normalization::zscore::defaultDense> >(
                           new normalization::zscore::Batch<algorithmFPType, normalization::zscore::defaultDense>()));

    services::SharedPtr<normalization::zscore::BatchImpl> normalization; /*!< Pointer to batch covariance */

    /**
                                                            * Checks batch parameter of the PCA svd algorithm
                                                            * \return Errors detected while checking
                                                            */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
    * <a name="DAAL-CLASS-ALGORITHMS__PCA__RESULT"></a>
    * \brief Provides methods to access results obtained with the PCA algorithm
    */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result(const Result & o);
    Result();

    virtual ~Result() {};

    /**
    * Gets the results of the PCA algorithm
        * \param[in] id    Identifier of the input object
        * \return          Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
    * Gets the results collection of the PCA algorithm
    * \param[in] id    Identifier of the results collection
    * \return          PCA results collection
    */
    data_management::KeyValueDataCollectionPtr get(ResultCollectionId id) const;

    /**
    * Sets the results collection of the PCA algorithm
    * only not NULL tables from collection collection will be set to result
    * \param[in] id          Identifier of the results collection
    * \param[in] collection  PCA results collection
    */
    void set(ResultCollectionId id, data_management::KeyValueDataCollectionPtr & collection);

    /**
        * Sets results of the PCA algorithm
        * \param[in] id      Identifier of the result
        * \param[in] value   Pointer to the object
        */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
        * Allocates memory for storing partial results of the PCA algorithm
        * \param[in] input Pointer to an object containing input data
        * \param[in] parameter Algorithm parameter
        * \param[in] method Computation method
        */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, daal::algorithms::Parameter * parameter, const Method method);

    /**
        * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
        * \param[in] parameter Parameter of the algorithm
        * \param[in] method        Computation method
        */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * partialResult, daal::algorithms::Parameter * parameter,
                                          const Method method);

    /**
    * Checks the results of the PCA algorithm
    * \param[in] _input  %Input object of algorithm
    * \param[in] par     Algorithm %parameter
    * \param[in] method  Computation  method
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::Input * _input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks the results of the PCA algorithm
    * \param[in] pr             Partial results of the algorithm
    * \param[in] method         Computation method
    * \param[in] parameter      Algorithm %parameter
    * \return Errors detected while checking
    */
    services::Status check(const daal::algorithms::PartialResult * pr, const daal::algorithms::Parameter * parameter,
                           int method) const DAAL_C11_OVERRIDE;

protected:
    /**
    * Checks the results of the PCA algorithm implementation
    * \param[in] nFeatures             Number of features
    * \param[in] nComponents           Number of components
    * \param[in] resultsToCompute      Results to compute
    *
    * \return Status
    */
    services::Status checkImpl(size_t nFeatures, size_t nComponents, DAAL_UINT64 resultsToCompute) const;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

private:
    Result & operator=(const Result &);
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface3
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResultBase;
using interface1::PartialResult;
using interface3::BatchParameter;
using interface3::BaseBatchParameter;
using interface1::OnlineParameter;
using interface1::DistributedParameter;
using interface1::DistributedInput;
using interface3::Result;
using interface3::ResultPtr;

} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
