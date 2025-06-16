/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/algo/pca/infer.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;

template <typename TestType>
class pca_serialization_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = task::dim_reduction;
    using descriptor_t = descriptor<float_t, method_t, task_t>;

    auto get_descriptor() {
        return descriptor_t{};
    }

    table get_data() {
        const std::int64_t row_count = 10;
        const std::int64_t column_count = 5;
        static const float_t data[] = {
            4.59,  0.81,  -1.37, -0.04, -0.75, //
            4.87,  0.34,  -0.98, 4.1,   -0.12, //
            4.44,  0.11,  -0.4,  3.27,  4.82, //
            0.59,  0.98,  -1.88, -0.64, 2.54, //
            -1.98, 2.57,  4.11,  -1.3,  -0.66, //
            3.26,  2.8,   2.65,  0.83,  2.12, //
            0.21,  4.23,  2.71,  2.2,   3.85, //
            1.27,  -1.15, 2.84,  1.11,  -1.12, //
            0.25,  1.61,  1.69,  4.51,  0.09, //
            -0.01, 0.58,  0.83,  2.73,  -1.33, //
        };
        return homogen_table::wrap(data, row_count, column_count);
    }

    model<task_t> train_model() {
        return this->train(this->get_descriptor(), this->get_data()).get_model();
    }

    infer_result<task_t> run_inference(const model<task_t>& m) {
        return this->infer(this->get_descriptor(), m, this->get_data());
    }

    void compare_infer_results(const infer_result<task_t>& actual,
                               const infer_result<task_t>& reference) {
        SECTION("compare responses") {
            te::check_if_tables_equal<float_t>(actual.get_transformed_data(),
                                               reference.get_transformed_data());
        }
    }

    void run_test() {
        INFO("training");
        const auto model = train_model();

        INFO("serialization");
        const auto deserialized_model = te::serialize_deserialize(model);

        INFO("inference");
        const auto expected = run_inference(model);
        const auto actual = run_inference(deserialized_model);
        compare_infer_results(actual, expected);
    }
};

using pca_types = COMBINE_TYPES((float, double), (method::cov));
TEMPLATE_LIST_TEST_M(pca_serialization_test,
                     "serialize/deserialize pca model",
                     "[pca]",
                     pca_types) {
    SKIP_IF(this->not_float64_friendly());
    this->run_test();
}

} // namespace oneapi::dal::pca::test
