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

#include "oneapi/dal/algo/decision_forest/parameters/cpu/train_parameters.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::decision_forest::parameters {

using task::classification;
using task::regression;

template struct ONEDAL_EXPORT train_parameters_cpu<float, method::hist, classification>;
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::dense, classification>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::hist, classification>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::dense, classification>;
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::hist, regression>;
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::dense, regression>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::hist, regression>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::dense, regression>;

} // namespace oneapi::dal::decision_forest::parameters
