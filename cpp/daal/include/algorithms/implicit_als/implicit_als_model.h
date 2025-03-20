/* file: implicit_als_model.h */
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
//  Declaration of the implicit ALS model class
//--
*/

#ifndef __IMPLICIT_ALS_MODEL_H__
#define __IMPLICIT_ALS_MODEL_H__

#include "algorithms/model.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup recommendation_systems Recommendation Systems
 * \brief Contains classes to work with recommendation systems
 * @ingroup training_and_prediction
 * @defgroup implicit_als Implicit Alternating Least Squares
 * \copydoc daal::algorithms::implicit_als
 * @ingroup recommendation_systems
 * @{
 */
namespace implicit_als
{
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__IMPLICIT_ALS__PARAMETER"></a>
 * \brief Parameters for the compute() method of the implicit ALS algorithm
 *
 * \snippet implicit_als/implicit_als_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     * Constructs parameters of the implicit ALS initialization algorithm
     * \param[in] nFactors            Number of factors
     * \param[in] maxIterations       Maximum number of iterations of the implicit ALS training algorithm
     * \param[in] alpha               Confidence parameter of the implicit ALS training algorithm
     * \param[in] lambda              Regularization parameter
     * \param[in] preferenceThreshold Threshold used to define preference values
     */
    Parameter(size_t nFactors = 10, size_t maxIterations = 5, double alpha = 40.0, double lambda = 0.01, double preferenceThreshold = 0.0)
        : nFactors(nFactors), maxIterations(maxIterations), alpha(alpha), lambda(lambda), preferenceThreshold(preferenceThreshold)
    {}

    size_t nFactors;            /*!< Number of factors */
    size_t maxIterations;       /*!< Maximum number of iterations of the implicit ALS training algorithm */
    double alpha;               /*!< Confidence parameter of the implicit ALS training algorithm */
    double lambda;              /*!< Regularization parameter */
    double preferenceThreshold; /*!< Threshold used to define preference values */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__MODEL"></a>
 * \brief Model trained by the implicit ALS algorithm in the batch processing mode
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - Parameter class
 */
class DAAL_EXPORT Model : public daal::algorithms::Model
{
public:
    DECLARE_MODEL(Model, daal::algorithms::Model);

    /**
     * Constructs the implicit ALS model
     * \param[in]  nUsers    Number of users in the input data set
     * \param[in]  nItems    Number of items in the input data set
     * \param[in]  parameter Implicit ALS parameters
     * \param[in]  dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nUsers, size_t nItems, const Parameter & parameter, modelFPType dummy);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    Model();

    /**
     * Constructs the implicit ALS model
     * \param[in]  nUsers    Number of users in the input data set
     * \param[in]  nItems    Number of items in the input data set
     * \param[in]  parameter Implicit ALS parameters
     * \param[out] stat      Status of the model construction
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(size_t nUsers, size_t nItems, const Parameter & parameter, services::Status * stat = NULL);

    virtual ~Model() {}

    /**
     * Returns a pointer to the numeric table of users factors constructed during the training
     * of the implicit ALS model
     * \return Numeric table of users factors
     */
    data_management::NumericTablePtr getUsersFactors() const { return _usersFactors; }

    /**
     * Returns a pointer to the numeric table of items factors constructed during the training
     * of the implicit ALS model
     * \return Numeric table of items factors
     */
    data_management::NumericTablePtr getItemsFactors() const { return _itemsFactors; }

private:
    data_management::NumericTablePtr _usersFactors; /* Table of resulting users factors */
    data_management::NumericTablePtr _itemsFactors; /* Table of resulting items factors */

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_usersFactors);
        arch->setSharedPtrObj(_itemsFactors);

        return services::Status();
    }

    template <typename modelFPType>
    DAAL_EXPORT Model(size_t nUsers, size_t nItems, const Parameter & parameter, modelFPType dummy, services::Status & st);
};
typedef services::SharedPtr<Model> ModelPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__PARTIALMODEL"></a>
 * \brief Partial model trained by the implicit ALS training algorithm in the distributed processing mode
 *
 * \par References
 *      - \ref training::interface1::Distributed "implicit_als::training::Distributed"
 *      - Parameter class
 */
class DAAL_EXPORT PartialModel : public daal::algorithms::Model
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialModel)

    /**
     * Constructs a partial implicit ALS model of a specified size
     * \param[in] parameter Implicit ALS parameters
     * \param[in] size      Model size
     * \param[in] dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT PartialModel(const Parameter & parameter, size_t size, modelFPType dummy);

    /**
     * Constructs a partial implicit ALS model from the indices of factors
     * \param[in] parameter Implicit ALS parameters
     * \param[in] offset    Index of the first factor in the partial model
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     * \param[in] dummy     Dummy variable for the templated constructor
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    template <typename modelFPType>
    DAAL_EXPORT PartialModel(const Parameter & parameter, size_t offset, data_management::NumericTablePtr indices, modelFPType dummy);

    /**
     * Constructs a partial implicit ALS model from the indices and factors stored in the numeric tables
     * \param[in] factors   Pointer to the numeric table with factors stored in row-major order
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    PartialModel(data_management::NumericTablePtr factors, data_management::NumericTablePtr indices);

    /**
     * Empty constructor for deserialization
     * \DAAL_DEPRECATED_USE{ Model::create }
     */
    PartialModel();

    /**
     * Constructs a partial implicit ALS model of a specified size
     * \param[in] parameter Implicit ALS parameters
     * \param[in] size      Model size
     * \param[out] stat     Status of the model construction
     * \return Partial implicit ALS model of a specified size
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<PartialModel> create(const Parameter & parameter, size_t size, services::Status * stat = NULL);
    /**
     * Constructs a partial implicit ALS model from the indices of factors
     * \param[in] parameter Implicit ALS parameters
     * \param[in] offset    Index of the first factor in the partial model
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     * \param[out] stat     Status of the model construction
     * \return Partial implicit ALS model with the specified indices and factors
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<PartialModel> create(const Parameter & parameter, size_t offset,
                                                                const data_management::NumericTablePtr & indices, services::Status * stat = NULL);
    /**
     * Constructs a partial implicit ALS model from the indices and factors stored in the numeric tables
     * \param[in] factors   Pointer to the numeric table with factors stored in row-major order
     * \param[in] indices   Pointer to the numeric table with the indices of factors
     * \param[out] stat     Status of the model construction
     * \return Partial implicit ALS model with the specified indices and factors
     */
    static services::SharedPtr<PartialModel> create(const data_management::NumericTablePtr & factors,
                                                    const data_management::NumericTablePtr & indices, services::Status * stat = NULL);

    virtual ~PartialModel() {}

    /**
     * Returns pointer to the numeric table with factors stored in row-major order
     * \return Pointer to the numeric table with factors stored in row-major order
     */
    data_management::NumericTablePtr getFactors() const { return _factors; }

    /**
     * Returns the pointer to the numeric table with the indices of factors
     * \return Pointer to the numeric table with the indices of factors
     */
    data_management::NumericTablePtr getIndices() const { return _indices; }

protected:
    data_management::NumericTablePtr _factors; /* Factors in row-major format */
    data_management::NumericTablePtr _indices; /* Indices of the factors */

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_factors);
        arch->setSharedPtrObj(_indices);

        return services::Status();
    }

    template <typename modelFPType>
    DAAL_EXPORT PartialModel(const Parameter & parameter, size_t size, modelFPType dummy, services::Status & st);

    template <typename modelFPType>
    DAAL_EXPORT PartialModel(const Parameter & parameter, size_t offset, const data_management::NumericTablePtr & indices, modelFPType dummy,
                             services::Status & st);

    PartialModel(const data_management::NumericTablePtr & factors, const data_management::NumericTablePtr & indices, services::Status & st);

private:
    template <typename modelFPType>
    DAAL_EXPORT services::Status initialize(const Parameter & parameter, size_t size);

    template <typename modelFPType>
    DAAL_EXPORT services::Status initialize(const Parameter & parameter, size_t offset, const data_management::NumericTablePtr & indices);
};

typedef services::SharedPtr<PartialModel> PartialModelPtr;
} // namespace interface1
using interface1::Parameter;
using interface1::ModelPtr;
using interface1::Model;
using interface1::PartialModelPtr;
using interface1::PartialModel;

} // namespace implicit_als
/** @} */
} // namespace algorithms
} // namespace daal

#endif
