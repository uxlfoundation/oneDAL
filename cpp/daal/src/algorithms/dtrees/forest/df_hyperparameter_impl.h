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
namespace internal
{

// The maximal value of the minPartCoefficient hyperparameter
constexpr DAAL_INT64 MAX_PART_COEFFICIENT = 256l;

// The maximal value of the minSizeCoefficient hyperparameter
constexpr DAAL_INT64 MAX_SIZE_COEFFICIENT = 0x100000l;

} // namespace internal

namespace classification
{
namespace training
{
namespace internal
{

// The maximal value of the number of classes supported by the implementation that is optimized
// for small number of classes
constexpr DAAL_INT64 MAX_SMALL_N_CLASSES = 8l;

/**
 * Available identifiers of integer hyperparameters of the decision forest training algorithm
 */
enum HyperparameterId
{
    smallNClassesThreshold = 0,
    minPartCoefficient     = 1,
    minSizeCoefficient     = 2,
    hyperparameterIdCount  = 3
};

enum DoubleHyperparameterId
{
    doubleHyperparameterIdCount = 0
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

    /**
     * Checks that all IDs are valid and the hyperparameter values are correct
     */
    void check(services::Status & s) const
    {
        DAAL_INT64 smallNClassesValue = 0l;
        DAAL_INT64 minPartCoeffValue  = 0l;
        DAAL_INT64 minSizeCoeffValue  = 0l;
        s |= find(smallNClassesThreshold, smallNClassesValue);
        s |= find(minPartCoefficient, minPartCoeffValue);
        s |= find(minSizeCoefficient, minSizeCoeffValue);
        if (!s) return;

        if (smallNClassesValue < 1 || MAX_SMALL_N_CLASSES < smallNClassesValue || minPartCoeffValue < 1
            || minPartCoeffValue > decision_forest::internal::MAX_PART_COEFFICIENT || minSizeCoeffValue < 1
            || minSizeCoeffValue > decision_forest::internal::MAX_SIZE_COEFFICIENT)
        {
            s.add(services::Error::create(services::ErrorHyperparameterBadValue));
        }
    }
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
    minPartCoefficient    = 0,
    minSizeCoefficient    = 1,
    hyperparameterIdCount = 2
};

enum DoubleHyperparameterId
{
    doubleHyperparameterIdCount = 0
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

    /**
     * Checks that all IDs are valid and the hyperparameter values are correct
     */
    void check(services::Status & s) const
    {
        DAAL_INT64 minPartCoeffValue = 0l;
        DAAL_INT64 minSizeCoeffValue = 0l;
        s |= find(minPartCoefficient, minPartCoeffValue);
        s |= find(minSizeCoefficient, minSizeCoeffValue);
        if (!s) return;

        if (minPartCoeffValue < 1 || minPartCoeffValue > decision_forest::internal::MAX_PART_COEFFICIENT || minSizeCoeffValue < 1
            || minSizeCoeffValue > decision_forest::internal::MAX_SIZE_COEFFICIENT)
        {
            s.add(services::Error::create(services::ErrorHyperparameterBadValue));
        }
    }
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
    blockSizeMultiplier              = 0,
    blockSize                        = 1,
    minTreesForThreading             = 2,
    minNumberOfRowsForVectSeqCompute = 3,
    hyperparameterIdCount            = 4
};

enum DoubleHyperparameterId
{
    scaleFactorForVectParallelCompute = 0,
    doubleHyperparameterIdCount       = 1
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
