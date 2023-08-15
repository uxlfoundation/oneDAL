/* file: covariance_hyperparameter.cpp */
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
//  Implementation of performance-related hyperparameters of the covariance algorithm.
//--
*/

#include "algorithms/covariance/covariance_types.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{

Hyperparameter::Hyperparameter() : algorithms::Hyperparameter(hyperparameterIdCount, doubleHyperparameterIdCount) {}

services::Status Hyperparameter::set(HyperparameterId id, std::int64_t value)
{
    return this->algorithms::Hyperparameter::set(std::uint32_t(id), value);
}

services::Status Hyperparameter::set(DoubleHyperparameterId id, double value)
{
    return this->algorithms::Hyperparameter::set(std::uint32_t(id), value);
}

services::Status Hyperparameter::find(HyperparameterId id, std::int64_t & value) const
{
    return this->algorithms::Hyperparameter::find(std::uint32_t(id), value);
}

services::Status Hyperparameter::find(DoubleHyperparameterId id, double & value) const
{
    return this->algorithms::Hyperparameter::find(std::uint32_t(id), value);
}

} // namespace interface1
} // namespace covariance
} // namespace algorithms
} // namespace daal
