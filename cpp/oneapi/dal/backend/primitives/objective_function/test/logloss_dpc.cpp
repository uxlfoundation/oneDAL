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

#include "fixture.hpp"

namespace oneapi::dal::backend::primitives::test {

TEMPLATE_LIST_TEST_M(logloss_test, "gold input test", "[logloss]", logloss_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input(this->fit_intercept_);
}

TEMPLATE_LIST_TEST_M(logloss_test, "test random input without L1", "[logloss]", logloss_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0f, 1.3f, this->fit_intercept_);
}

TEMPLATE_LIST_TEST_M(logloss_test, "batch test", "[logloss]", logloss_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0f, 1.3f, this->fit_intercept_, true);
}

TEMPLATE_LIST_TEST_M(logloss_test, "test random input with L1", "[logloss]", logloss_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4f, 1.3f, this->fit_intercept_);
}

TEMPLATE_LIST_TEST_M(logloss_test, "sparse data test without L2", "[logloss]", logloss_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(!te::device_sparse_blas_supported());
    this->generate_sparse_input();
    this->run_sparse_test(0.0f, this->fit_intercept_);
}

TEMPLATE_LIST_TEST_M(logloss_test, "sparse data test", "[logloss]", logloss_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(!te::device_sparse_blas_supported());
    this->generate_sparse_input();
    this->run_sparse_test(1.3f, this->fit_intercept_);
}

} // namespace oneapi::dal::backend::primitives::test
