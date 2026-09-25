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

#include "oneapi/dal/algo/basic_statistics/test/fixture.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;
namespace bs = oneapi::dal::basic_statistics;

using online_csr_types = basic_statistics_online_csr_types;

TEMPLATE_LIST_TEST_M(basic_statistics_online_csr_test,
                     "basic_statistics online CSR flow",
                     "[basic_statistics][integration][online]",
                     online_csr_types) {
    SKIP_IF(this->not_float64_friendly());
    const float nnz_fraction = 0.05;
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const auto data =
        GENERATE_COPY(te::csr_table_builder(20, 10, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 20, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 100, nnz_fraction, this->data_indexing_));
    SKIP_IF(this->not_cpu_friendly(data));

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);
    const std::int64_t n_blocks = GENERATE(1, 3, 5);

    this->online_csr_general_checks(data, compute_mode, n_blocks);
}

TEMPLATE_LIST_TEST_M(basic_statistics_online_csr_test,
                     "basic_statistics online CSR flow with an all-zero batch",
                     "[basic_statistics][integration][online]",
                     online_csr_types) {
    SKIP_IF(this->not_float64_friendly());
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const bool zeros_first = GENERATE(true, false);

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));
    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_all);

    this->online_csr_zero_batch_checks(compute_mode, zeros_first);
}

TEMPLATE_LIST_TEST_M(basic_statistics_online_csr_test,
                     "basic_statistics online CSR flow with weights",
                     "[basic_statistics][integration][online]",
                     online_csr_types) {
    SKIP_IF(this->not_float64_friendly());
    const float nnz_fraction = 0.05;
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const auto data =
        GENERATE_COPY(te::csr_table_builder(20, 10, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 20, nnz_fraction, this->data_indexing_));
    SKIP_IF(this->not_cpu_friendly(data));

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);
    const std::int64_t n_blocks = GENERATE(1, 3);

    this->online_csr_weighted_checks(data, compute_mode, n_blocks);
}

/// Weights that straddle zero, checked on the `min | max` mask specifically. A negative
/// weight flips the sign of its row's contribution, so it moves entries from the maximum
/// of a column to the minimum and back. Only a path that really folds the weights into
/// the values -- implicit zeros and all -- reproduces the dense weighted extrema here.
/// Restricted to `min | max` on purpose: mixed-sign weights drive column means towards
/// zero, and `variation = stddev / mean` is not meaningfully comparable there.
TEMPLATE_LIST_TEST_M(basic_statistics_online_csr_test,
                     "basic_statistics online CSR flow with sign-changing weights",
                     "[basic_statistics][integration][online]",
                     online_csr_types) {
    SKIP_IF(this->not_float64_friendly());
    const float nnz_fraction = 0.05;
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const auto data =
        GENERATE_COPY(te::csr_table_builder(20, 10, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 20, nnz_fraction, this->data_indexing_));
    SKIP_IF(this->not_cpu_friendly(data));

    const bs::result_option_id compute_mode = result_options::min | result_options::max;
    const std::int64_t n_blocks = GENERATE(1, 3);

    this->online_csr_weighted_checks(data, compute_mode, n_blocks, -2.5, 2.5);
}

} // namespace oneapi::dal::basic_statistics::test
