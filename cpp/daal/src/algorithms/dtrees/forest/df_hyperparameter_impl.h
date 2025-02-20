/* file: df_hyperparameter_impl.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
//  Declaration of the class that implements performance-related hyperparameters
//  of the decision forest algorithm.
//--
*/

#ifndef __DF_HYPERPARAMETER_IMPL_H__
#define __DF_HYPERPARAMETER_IMPL_H__

#include "algorithms/algorithm_types.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{

/**
 * Available identifiers of integer hyperparameters of the decision forest training algorithm
 */
enum HyperparameterId
{
    smallNClassesThreshold = 0,
    minPartCoefficient,
    minSizeCoefficient,
    hyperparameterIdCount = minSizeCoefficient + 1
};

enum DoubleHyperparameterId
{
    doubleHyperparameterIdCount       = 0
};

/**
 * \brief Hyperparameters of the decision forest training algorithm
 */
struct DAAL_EXPORT Hyperparameter : public daal::algorithms::Hyperparameter
{
    using algorithms::Hyperparameter::set;
    using algorithms::Hyperparameter::find;

    /** Default constructor */
    Hyperparameter();

    /**
     * Sets integer hyperparameter of the decision forest algorithm
     * \param[in] id        Identifier of the hyperparameter
     * \param[in] value     The value of the hyperparameter
     */
    services::Status set(HyperparameterId id, DAAL_INT64 value);

    /**
     * Sets double precision hyperparameter of the decision forest algorithm
     * \param[in] id        Identifier of the hyperparameter
     * \param[in] value     Value of the hyperparameter
     */
    services::Status set(DoubleHyperparameterId id, double value);

    /**
     * Finds integer hyperparameter of the decision forest algorithm by its identifier
     * \param[in]  id       Identifier of the hyperparameter
     * \param[out] value    Value of the found hyperparameter
     */
    services::Status find(HyperparameterId id, DAAL_INT64 & value) const;

    /**
     * Finds double precision hyperparameter of the decision forest algorithm by its identifier
     * \param[in]  id       Identifier of the hyperparameter
     * \param[out] value    Value of the found hyperparameter
     */
    services::Status find(DoubleHyperparameterId id, double & value) const;
};

} // namespace internal
} // namespace training
} // namespace classification

namespace regression
{
namespace training
{
namespace internal
{

/**
 * Available identifiers of integer hyperparameters of the decision forest training algorithm
 */
enum HyperparameterId
{
    minPartCoefficient = 0,
    minSizeCoefficient,
    hyperparameterIdCount = minSizeCoefficient + 1
};

enum DoubleHyperparameterId
{
    doubleHyperparameterIdCount       = 0
};

/**
 * \brief Hyperparameters of the decision forest training algorithm
 */
struct DAAL_EXPORT Hyperparameter : public daal::algorithms::Hyperparameter
{
    using algorithms::Hyperparameter::set;
    using algorithms::Hyperparameter::find;

    /** Default constructor */
    Hyperparameter();

    /**
     * Sets integer hyperparameter of the decision forest algorithm
     * \param[in] id        Identifier of the hyperparameter
     * \param[in] value     The value of the hyperparameter
     */
    services::Status set(HyperparameterId id, DAAL_INT64 value);

    /**
     * Sets double precision hyperparameter of the decision forest algorithm
     * \param[in] id        Identifier of the hyperparameter
     * \param[in] value     Value of the hyperparameter
     */
    services::Status set(DoubleHyperparameterId id, double value);

    /**
     * Finds integer hyperparameter of the decision forest algorithm by its identifier
     * \param[in]  id       Identifier of the hyperparameter
     * \param[out] value    Value of the found hyperparameter
     */
    services::Status find(HyperparameterId id, DAAL_INT64 & value) const;

    /**
     * Finds double precision hyperparameter of the decision forest algorithm by its identifier
     * \param[in]  id       Identifier of the hyperparameter
     * \param[out] value    Value of the found hyperparameter
     */
    services::Status find(DoubleHyperparameterId id, double & value) const;
};

} // namespace internal
} // namespace training
} // namespace regression

namespace prediction
{
namespace internal
{

/**
 * Available identifiers of integer hyperparameters of the decision forest prediction algorithm
 */
enum HyperparameterId
{
    blockSizeMultiplier = 0,
    blockSize,
    minTreesForThreading,
    minNumberOfRowsForVectSeqCompute,
    hyperparameterIdCount = minNumberOfRowsForVectSeqCompute + 1
};

enum DoubleHyperparameterId
{

    scaleFactorForVectParallelCompute = 0,
    doubleHyperparameterIdCount       = scaleFactorForVectParallelCompute + 1
};

/**
 * \brief Hyperparameters of the decision forest prediction algorithm
 */
struct DAAL_EXPORT Hyperparameter : public daal::algorithms::Hyperparameter
{
    using algorithms::Hyperparameter::set;
    using algorithms::Hyperparameter::find;

    /** Default constructor */
    Hyperparameter();

    /**
     * Sets integer hyperparameter of the decision forest algorithm
     * \param[in] id        Identifier of the hyperparameter
     * \param[in] value     The value of the hyperparameter
     */
    services::Status set(HyperparameterId id, DAAL_INT64 value);

    /**
     * Sets double precision hyperparameter of the decision forest algorithm
     * \param[in] id        Identifier of the hyperparameter
     * \param[in] value     Value of the hyperparameter
     */
    services::Status set(DoubleHyperparameterId id, double value);

    /**
     * Finds integer hyperparameter of the decision forest algorithm by its identifier
     * \param[in]  id       Identifier of the hyperparameter
     * \param[out] value    Value of the found hyperparameter
     */
    services::Status find(HyperparameterId id, DAAL_INT64 & value) const;

    /**
     * Finds double precision hyperparameter of the decision forest algorithm by its identifier
     * \param[in]  id       Identifier of the hyperparameter
     * \param[out] value    Value of the found hyperparameter
     */
    services::Status find(DoubleHyperparameterId id, double & value) const;
};

} // namespace internal
} // namespace prediction
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif // __DF_HYPERPARAMETER_IMPL_H__
