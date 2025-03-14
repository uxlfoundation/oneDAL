/* file: pca_transform_types.h */
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
//  Definition of PCA transform common types.
//--
*/

#ifndef __PCA_TRANSFORM_TYPES_H__
#define __PCA_TRANSFORM_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
/**
 * @defgroup pca_transform PCA Transformation
 * \copydoc daal::algorithms::pca::transform
 * @ingroup pca
 * @{
 */
/** \brief Contains classes for computing the results of the PCA transformation algorithm */

namespace transform
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__TRANSFORM__METHOD"></a>
 * Available methods for computing the PCA transformation algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__TRANSFORM__INPUTID"></a>
 * Available types of input objects for the PCA transformation algorithm
 */
enum InputId
{
    data = 0,     /*!< Input data table */
    eigenvectors, /*!< Transformation matrix of eigenvectors */
    lastInputId = eigenvectors
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__TRANSFORM__PCAID"></a>
 * Available types of dataForTransform - TODO: remove after extending PCA
 */
enum TransformComponentId
{
    mean       = 0x00000001ULL, /*!< Numeric table of size 1 x p with the mean values of features >*/
    variance   = 0x00000002ULL, /*!< Numeric table of size 1 x p with the variances of features >*/
    eigenvalue = 0x00000004ULL  /*!< Numeric table of size 1 x p with the always computed eigenvalues>*/
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__TRANSFORM__TRANSFORMDATAINPUTID"></a>
 * Available types of input objects for the PCA transformation algorithm
 */
enum TransformDataInputId
{
    dataForTransform            = lastInputId + 1, /*!< Data for transform */
    lastdataForTransformInputId = dataForTransform
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PCA__TRANSFORM__RESULTID"></a>
 * Available types of results of the PCA transformation algorithm
 */
enum ResultId
{
    transformedData = 0, /*!< Transformed data */
    lastResultId    = transformedData
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__INPUT"></a>
 * \brief Input objects for the PCA transformation algorithm in the batch and online processing modes and for the first distributed step of the
 * algorithm.
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other);

    /** Destructor */
    virtual ~Input() {}

    /**
     * Returns input object of the PCA transformation algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Returns input object of the PCA transformation algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(TransformDataInputId id) const;

    /**
     * Returns input transform object of the PCA transformation algorithm
     * \param[in] wid   Identifier of the transform data input object
     * \param[in] id    Identifier of the transform data contained object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(TransformDataInputId wid, TransformComponentId id) const;

    /**
     * Sets input object for the PCA transformation algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    /**
     * Sets input object for the PCA transformation algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(TransformDataInputId id, const data_management::KeyValueDataCollectionPtr & value);

    /**
     * Sets input transform object for the PCA transformation algorithm
     * \param[in] wid   Identifier of the transform data input object
     * \param[in] id    Identifier of the transform data contained object
     * \param[in] value Pointer to the input object
     */
    void set(TransformDataInputId wid, TransformComponentId id, const data_management::NumericTablePtr & value);

    /**
     * Checks parameters of the algorithm
     * \param[in] parameter Pointer to the parameters
     * \param[in] method Computation method
     */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the PCA transformation algorithm
 *        in the batch processing mode or finalizeCompute() method of algorithm in the online processing mode
 *        or on the second and third steps of the algorithm in the distributed processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    /** Default constructor */
    Result();
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Returns the result of the PCA transformation algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Allocates memory for storing final results of the PCA transformation algorithm
     * \param[in] input     Pointer to input object
     * \param[in] parameter Pointer to parameter
     * \param[in] method    Algorithm method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Sets an input object for the PCA transformation algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Checks final results of the algorithm
     * \param[in] input  Pointer to input objects
     * \param[in] par    Pointer to parameters
     * \param[in] method Computation method
     */
    virtual services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                   int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;
};
typedef services::SharedPtr<daal::algorithms::pca::transform::interface1::Result> ResultPtr;

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__PCA__TRANSFORM__PARAMETER"></a>
 * \brief Parameters for the PCA transformation compute method
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Parameter constructor
     * \param[in] nComponents Number of principal components
     */
    Parameter(size_t nComponents = 0);

    /**
     *  Number of components
     */
    size_t nComponents;
};
/** @} */
/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::Parameter;

} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
