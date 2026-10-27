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

#include "oneapi/dal/backend/primitives/objective_function/test/spmd_fixture.hpp"

namespace oneapi::dal::backend::primitives::test {

using logloss_spmd_types = COMBINE_TYPES((float, double), (use_fit_intercept));

TEMPLATE_LIST_TEST_M(logloss_spmd_test,
                     "spmd test - double",
                     "[logloss spmd]",
                     logloss_spmd_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_spmd(-1, 1.0, true);
    this->run_spmd(-1, 1.0, false);
}

} // namespace oneapi::dal::backend::primitives::test
