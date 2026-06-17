/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "oneapi/dal/algo/knn/test/fixture.hpp"
#include "oneapi/dal/detail/cpu.hpp" // detect_cpu_features

#include "daal/src/services/service_defines.h" // daal_set_float32_matmul_precision

namespace oneapi::dal::knn::test {

template <typename TestType>
class knn_batch_test : public knn_test<TestType, knn_batch_test<TestType>> {};

TEMPLATE_LIST_TEST_M(knn_batch_test,
                     "knn nearest points test require_bf16",
                     "[synthetic-dataset][knn][integration][batch][test]",
                     knn_bf_cls_float_only) {
    SKIP_IF(this->get_policy().is_gpu());
    /// Check if AMX-BF16 is supported and enabled on the system
    SKIP_IF(!(oneapi::dal::detail::detect_cpu_features() &
              uint64_t(oneapi::dal::detail::cpu_feature::amx_bf16)));

    constexpr std::int64_t train_row_count = 513;
    constexpr std::int64_t infer_row_count = 317;
    constexpr std::int64_t column_count = 17;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    daal_set_float32_matmul_precision(daal::internal::Float32MatmulPrecision::require_bf16);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);

    daal_set_float32_matmul_precision(daal::internal::Float32MatmulPrecision::strict);
}

} // namespace oneapi::dal::knn::test
