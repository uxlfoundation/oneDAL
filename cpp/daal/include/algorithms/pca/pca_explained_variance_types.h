/* file: pca_explained_variance_types.h */
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
//  Interface for the PCA algorithm quality metrics for a explained variance
//--
*/

#ifndef __PCA_QUALITY_METRIC_TYPES_H__
#define __PCA_QUALITY_METRIC_TYPES_H__

#include "services/daal_shared_ptr.h"
#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace quality_metric
{
/**
 * @defgroup pca_quality_metric_explained_variance explained variance
 * \copydoc daal::algorithms::pca::quality_metric_set::explained_variance
 * @ingroup pca_quality_metric_set
 * @{
 */
namespace explained_variance
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINED_VARIANCE__METHOD"></a>
 * Available methods for computing the quality metrics for a explained variance
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINED_VARIANCE__INPUTID"></a>
* \brief Available identifiers of input objects for a explained variance quality metrics
*/
enum InputId
{
    eigenvalues, /*!< NumericTable 1 x k. Eigenvalues of PCA */
    lastInputId = eigenvalues
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINED_VARIANCE__RESULTID"></a>
* \brief Available identifiers of the result of explained variance quality metrics
*/
enum ResultId
{
    explainedVariances,       /*!< NumericTable 1 x k. Explained variances */
    explainedVariancesRatios, /*!< NumericTable 1 x k. Explained variances ratios */
    noiseVariance,            /*!< NumericTable 1 x 1. Noise variance */
    lastResultId = noiseVariance
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINED_VARIANCE__PARAMETER"></a>
 * \brief Parameters for the compute() method of explained variance quality metrics
 *
 * \snippet pca/pca_explained_variance_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public daal::algorithms::Parameter
{
    DAAL_EXPORT Parameter(size_t nFeatures, size_t nComponents);
    virtual ~Parameter() {}

    size_t nFeatures;   /*!< Number of features */
    size_t nComponents; /*!< Number of components*/

    /**
    * Checks the correctness of the parameter
    *
    * \return Status of computations
    */
    DAAL_EXPORT virtual services::Status check() const;
};
/* [Parameter source code] */

/**
* <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINED_VARIANCE__INPUT"></a>
* \brief %Input objects for explained variance quality metrics
*/
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    DAAL_CAST_OPERATOR(Input)
    DAAL_DOWN_CAST_OPERATOR(Input, daal::algorithms::Input)

    /** Default constructor */
    Input();

    virtual ~Input() {}

    /**
    * Returns an input object for linear regression quality metric
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets an input object for linear regression quality metric
    * \param[in] id      Identifier of the input object
    * \param[in] value   Pointer to the object
    */
    void set(InputId id, const data_management::NumericTablePtr & value);

    /**
    * Checks an input object for the linear regression algorithm
    * \param[in] par     Algorithm parameter
    * \param[in] method  Computation method
    *
    * \return Status of computations
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};
typedef services::SharedPtr<Input> InputPtr;

/**
* <a name="DAAL-CLASS-ALGORITHMS__PCA__QUALITY_METRIC__EXPLAINED_VARIANCE__RESULT"></a>
* \brief Provides interface for the result of linear regression quality metrics
*/
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    DAAL_DOWN_CAST_OPERATOR(Result, daal::algorithms::Result)

    Result();

    /**
    * Returns the result of linear regression quality metrics
    * \param[in] id    Identifier of the result
    * \return          Result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
    * Sets the result of linear regression quality metrics
    * \param[in] id      Identifier of the input object
    * \param[in] value   %Input object
    */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Allocates memory to store
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Algorithm method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Checks the result of linear regression quality metrics
     * \param[in] input   %Input object
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
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

} // namespace interface1
using interface1::Parameter;
using interface1::Result;
using interface1::ResultPtr;
using interface1::Input;
using interface1::InputPtr;

} // namespace explained_variance
/** @} */
} // namespace quality_metric
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif // __PCA_QUALITY_METRIC_TYPES_H__
