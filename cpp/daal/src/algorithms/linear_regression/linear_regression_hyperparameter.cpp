/* file: linear_regression_hyperparameter.cpp */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//  Implementation of performance-related hyperparameters of the linear_regression algorithm.
//--
*/

#include <stdint.h>
#include "src/algorithms/linear_model/linear_model_hyperparameter_impl.h"
#include "src/algorithms/linear_regression/linear_regression_hyperparameter_impl.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace internal
{

Hyperparameter::Hyperparameter() : algorithms::Hyperparameter(hyperparameterIdCount, doubleHyperparameterIdCount) {}

services::Status Hyperparameter::set(HyperparameterId id, DAAL_INT64 value)
{
    return this->algorithms::Hyperparameter::set(uint32_t(id), value);
}

services::Status Hyperparameter::set(DoubleHyperparameterId id, double value)
{
    return this->algorithms::Hyperparameter::set(uint32_t(id), value);
}

services::Status Hyperparameter::find(HyperparameterId id, DAAL_INT64 & value) const
{
    return this->algorithms::Hyperparameter::find(uint32_t(id), value);
}

services::Status Hyperparameter::find(DoubleHyperparameterId id, double & value) const
{
    return this->algorithms::Hyperparameter::find(uint32_t(id), value);
}

services::Status convert(const Hyperparameter * params, services::SharedPtr<LinearModelHyperparameter> & result)
{
    typedef linear_model::internal::HyperparameterId LinearModelHyperparameterId;

    services::Status st;

    if (params != nullptr)
    {
        DAAL_INT64 denseUpdateStepBlockSize     = 0l;
        DAAL_INT64 denseUpdateMaxColsBatched    = 0l;
        DAAL_INT64 denseSmallRowsThreshold      = 0l;
        DAAL_INT64 denseSmallRowsMaxColsBatched = 0l;
        DAAL_INT64 denseGrainSize               = 0l;

        auto * const resultPtr = new LinearModelHyperparameter();
        DAAL_CHECK_MALLOC(resultPtr);
        result.reset(resultPtr);

        /// Getters
        st |= params->find(HyperparameterId::denseUpdateStepBlockSize, denseUpdateStepBlockSize);
        DAAL_CHECK(st, services::ErrorHyperparameterNotFound);

        st |= params->find(HyperparameterId::denseUpdateMaxColsBatched, denseUpdateMaxColsBatched);
        DAAL_CHECK(st, services::ErrorHyperparameterNotFound);

        st |= params->find(HyperparameterId::denseSmallRowsThreshold, denseSmallRowsThreshold);
        DAAL_CHECK(st, services::ErrorHyperparameterNotFound);

        st |= params->find(HyperparameterId::denseSmallRowsMaxColsBatched, denseSmallRowsMaxColsBatched);
        DAAL_CHECK(st, services::ErrorHyperparameterNotFound);

        st |= params->find(HyperparameterId::denseGrainSize, denseGrainSize);
        DAAL_CHECK(st, services::ErrorHyperparameterNotFound);

        /// Setters
        st |= result->set(LinearModelHyperparameterId::denseUpdateStepBlockSize, denseUpdateStepBlockSize);
        DAAL_CHECK(st, services::ErrorHyperparameterCanNotBeSet);

        st |= result->set(LinearModelHyperparameterId::denseUpdateMaxColsBatched, denseUpdateMaxColsBatched);
        DAAL_CHECK(st, services::ErrorHyperparameterCanNotBeSet);

        st |= result->set(LinearModelHyperparameterId::denseSmallRowsThreshold, denseSmallRowsThreshold);
        DAAL_CHECK(st, services::ErrorHyperparameterCanNotBeSet);

        st |= result->set(LinearModelHyperparameterId::denseSmallRowsMaxColsBatched, denseSmallRowsMaxColsBatched);
        DAAL_CHECK(st, services::ErrorHyperparameterCanNotBeSet);

        st |= result->set(LinearModelHyperparameterId::denseGrainSize, denseGrainSize);
        DAAL_CHECK(st, services::ErrorHyperparameterCanNotBeSet);
    }
    else
    {
        result.reset((LinearModelHyperparameter *)nullptr);
    }

    return st;
}

} // namespace internal
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
