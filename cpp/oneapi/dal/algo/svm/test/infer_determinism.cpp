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

#include <random>
#include <vector>

#include "oneapi/dal/algo/svm/infer.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::svm::test {

namespace te = dal::test::engine;

template <typename TestType>
class svm_infer_determinism_test
        : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;

    table make_random_table(std::int64_t row_count,
                            std::int64_t column_count,
                            float_t min,
                            float_t max,
                            std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float_t> dist(min, max);
        auto data = array<float_t>::empty(row_count * column_count);
        float_t* ptr = data.get_mutable_data();
        for (std::int64_t i = 0; i < row_count * column_count; ++i) {
            ptr[i] = dist(rng);
        }
        return homogen_table::wrap(data, row_count, column_count);
    }
};

using svm_infer_determinism_types = COMBINE_TYPES((float, double),
                                                  (svm::task::regression,
                                                   svm::task::classification));

TEMPLATE_LIST_TEST_M(svm_infer_determinism_test,
                     "repeated RBF inference with many support vectors is bitwise stable",
                     "[svm][infer][determinism]",
                     svm_infer_determinism_types) {
    SKIP_IF(this->get_policy().is_gpu());
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;
    using task_t = std::tuple_element_t<1, TestType>;
    using kernel_t = dal::rbf_kernel::descriptor<float_t>;
    using descriptor_t = descriptor<float_t, method::thunder, task_t, kernel_t>;

    // Many row blocks and support vectors; on non-Intel CPUs the RBF kernel then runs its own
    // parallel loop inside each predict block.
    constexpr std::int64_t row_count = 2048;
    constexpr std::int64_t sv_count = 12000;
    constexpr std::int64_t feature_count = 58;
    constexpr std::int64_t repeat_count = 8;

    const table x = this->make_random_table(row_count, feature_count, -1.0, 1.0, 7777);
    const table sv = this->make_random_table(sv_count, feature_count, -1.0, 1.0, 8888);
    const table coeffs = this->make_random_table(sv_count, 1, -50.0, 50.0, 9999);
    const table biases = this->make_random_table(1, 1, 0.0, 1.0, 1111);

    auto model = svm::model<task_t>{}.set_support_vectors(sv).set_coeffs(coeffs).set_biases(biases);
    constexpr bool is_regression = std::is_same_v<task_t, svm::task::regression>;
    if constexpr (!is_regression) {
        model.set_first_class_response(0).set_second_class_response(1);
    }

    const auto desc = descriptor_t{ kernel_t{}.set_sigma(5.0) };

    const auto predict = [&]() {
        const auto result = this->infer(desc, model, x);
        if constexpr (is_regression) {
            return row_accessor<const float_t>(result.get_responses()).pull({ 0, -1 });
        }
        else {
            return row_accessor<const float_t>(result.get_decision_function()).pull({ 0, -1 });
        }
    };

    const auto reference = predict();

    for (std::int64_t r = 0; r < repeat_count; ++r) {
        const auto values = predict();
        REQUIRE(values.get_count() == reference.get_count());
        std::int64_t mismatch_count = 0;
        for (std::int64_t i = 0; i < values.get_count(); ++i) {
            mismatch_count += values[i] != reference[i];
        }
        INFO("repeat " << r);
        REQUIRE(mismatch_count == 0);
    }
}

} // namespace oneapi::dal::svm::test
