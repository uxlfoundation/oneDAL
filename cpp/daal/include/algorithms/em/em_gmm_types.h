/* file: em_gmm_types.h */
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
//  Implementation of the EM for GMM algorithm interface.
//--
*/

#ifndef __EM_GMM_TYPES_H__
#define __EM_GMM_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/covariance/covariance_batch.h"
#include "algorithms/em/em_gmm_covariance_storage_id.h"
#include "algorithms/em/em_gmm_init_types.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup em_gmm_compute Computation
 * \copydoc daal::algorithms::em_gmm
 * @ingroup em_gmm
 * @{
 */
/**
 * \brief Contains classes for the EM for GMM algorithm
 */
namespace em_gmm
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__METHOD"></a>
 * Available methods for computing results of the EM for GMM algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__INPUTID"></a>
 * Available identifiers of input objects of the EM for GMM algorithm
 */
enum InputId
{
    data,         /*!< %Input data table */
    inputWeights, /*!< Input weights */
    inputMeans,   /*!< Input means */
    lastInputId = inputMeans
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__INPUTCOVARIANCESID"></a>
 * Available identifiers of input covariances for the EM for GMM algorithm
 */
enum InputCovariancesId
{
    inputCovariances       = lastInputId + 1, /*!< %Collection of input covariances */
    lastInputCovariancesId = inputCovariances
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__INPUTVALUESID"></a>
 * Available identifiers of input values for the EM for GMM algorithm
 */
enum InputValuesId
{
    inputValues       = lastInputCovariancesId + 1, /*!< Input objects of the EM for GMM algorithm */
    lastInputValuesId = inputValues
};

/**
 * <a name="DAAL-ENUM-EM_GMM__RESULTID"></a>
 * Available identifiers of results (means or weights) of the EM for GMM algorithm
 */
enum ResultId
{
    weights,      /*!< Weights */
    means,        /*!< Means */
    goalFunction, /*!< Table containing log-likelihood value */
    nIterations,  /*!< Table containing the number of executed iterations */
    lastResultId = nIterations
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__EM_GMM__RESULTCOVARIANCESID"></a>
 * Available identifiers of computed covariances for the EM for GMM algorithm
 */
enum ResultCovariancesId
{
    covariances             = lastResultId + 1, /*!< %Collection of covariances */
    lastResultCovariancesId = covariances
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__EM_GMM__PARAMETER"></a>
 * \brief %Parameter for the EM for GMM algorithm
 *
 * \snippet em/em_gmm_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs the parameter of EM for GMM algorithm
     * \param[in] nComponents              Number of components in the Gaussian mixture model
     * \param[in] covariance               Pointer to the algorithm that computes the covariance
     * \param[in] maxIterations            Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold        Threshold for the termination of the algorithm
     * \param[in] regularizationFactor     Factor for covariance regularization in case of ill-conditional data
     * \param[in] covarianceStorage        Type of covariance in the Gaussian mixture model.
     */
    Parameter(const size_t nComponents, const services::SharedPtr<covariance::BatchImpl> & covariance, const size_t maxIterations = 10,
              const double accuracyThreshold = 1.0e-04, const double regularizationFactor = 0.01, const CovarianceStorageId covarianceStorage = full);

    Parameter(const Parameter & other);

    virtual ~Parameter() {}

    /**
     * Checks the correctness of the parameter
     */
    virtual services::Status check() const;

    size_t nComponents;                                    /*!< Number of components in the Gaussian mixture model */
    size_t maxIterations;                                  /*!< Maximal number of iterations of the algorithm. */
    double accuracyThreshold;                              /*!< Threshold for the termination of the algorithm.    */
    services::SharedPtr<covariance::BatchImpl> covariance; /*!< Pointer to the algorithm that computes the covariance */
    double regularizationFactor;                           /*!< Factor for covariance regularization in case of ill-conditional data */
    CovarianceStorageId covarianceStorage;                 /*!< Type of covariance in the Gaussian mixture model. */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__INPUT"></a>
 * \brief %Input objects for the EM for GMM algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other) : daal::algorithms::Input(other) {}

    virtual ~Input() {}

    /**
     * Sets one input object for the EM for GMM algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the input covariance object for the EM for GMM algorithm
     * \param[in] id    Identifier of the input covariance collection object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputCovariancesId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Sets input objects for the EM for GMM algorithm
     * \param[in] id    Identifier of the input values object. Result of the EM for GMM initialization algorithm can be used.
     * \param[in] ptr   Pointer to the object
     */
    void set(InputValuesId id, const init::ResultPtr & ptr);

    /**
     * Returns the input numeric table for the EM for GMM algorithm
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Returns the collection of input covariances for the EM for GMM algorithm
     * \param[in] id    Identifier of the  collection of input covariances
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(InputCovariancesId id) const;

    /**
     * Returns a covariance with a given index from the collection of input covariances
     * \param[in] id    Identifier of the collection of input covariances
     * \param[in] index Index of the covariance to be returned
     * \return          Pointer to the table with the input covariance
     */
    data_management::NumericTablePtr get(InputCovariancesId id, size_t index) const;

    /**
     * Checks the correctness of the input result
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the EM for GMM algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    /** Default constructor */
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory for storing results of the EM for GMM algorithm
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Sets the result of the EM for GMM algorithm
     * \param[in] id    %Result identifier
     * \param[in] ptr   Pointer to the numeric table with the result
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets the collection of covariances for the EM for GMM algorithm
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] ptr   Pointer to the collection of covariances
     */
    void set(ResultCovariancesId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Returns the result of the EM for GMM algorithm
     * \param[in] id   %Result identifier
     * \return         %Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Returns the collection of computed covariances of the EM for GMM algorithm
     * \param[in] id   Identifier of the collection of computed covariances
     * \return         Collection of computed covariances that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(ResultCovariancesId id) const;

    /**
     * Returns the covariance with a given index from the collection of computed covariances
     * \param[in] id    Identifier of the collection of covariances
     * \param[in] index Index of the covariance to be returned
     * \return          Pointer to the table with the computed covariance
     */
    data_management::NumericTablePtr get(ResultCovariancesId id, size_t index) const;

    /**
    * Checks the result parameter of the EM for GMM algorithm
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace em_gmm
} // namespace algorithms
} // namespace daal
#endif
