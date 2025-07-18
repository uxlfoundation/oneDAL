/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/linear_regression/test/spmd_backend_template.hpp"

namespace oneapi::dal::linear_regression::test {

TEMPLATE_LIST_TEST_M(lr_spmd_test, "LR common flow", "[lr][spmd]", lr_types) {
    SKIP_IF(this->not_float64_friendly());

    this->generate(777);
    this->set_rank_count(GENERATE(2, 3));

    this->run_and_check_linear();
}

} // namespace oneapi::dal::linear_regression::test
