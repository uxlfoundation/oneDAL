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

#include "oneapi/dal/algo/basic_statistics/test/fixture.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace bs = oneapi::dal::basic_statistics;

template <typename TestType>
class basic_statistics_batch_test
        : public basic_statistics_test<TestType, basic_statistics_batch_test<TestType>> {};

TEMPLATE_LIST_TEST_M(basic_statistics_batch_test,
                     "basic_statistics common flow",
                     "[basic_statistics][integration][batch]",
                     basic_statistics_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 2, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 10, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 100, 10 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 200, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 500, 250 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 6000, 20 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 6000, 530 }.fill_normal(-30, 30, 7777),
                           te::dataframe_builder{ 10000, 20 }.fill_normal(-0.5, 0.5, 7777));

    std::shared_ptr<te::dataframe> weights;
    const bool use_weights = GENERATE(0, 1);

    if (use_weights) {
        const auto row_count = data.get_row_count();
        weights = std::make_shared<te::dataframe>(
            te::dataframe_builder{ row_count, 1 }.fill_normal(0, 1, 777).build());
    }

    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);

    this->general_checks(data, weights, compute_mode);
}

TEMPLATE_LIST_TEST_M(basic_statistics_batch_test,
                     "basic_statistics common CSR flow",
                     "[basic_statistics][integration][batch]",
                     basic_statistics_sparse_types) {
    SKIP_IF(this->not_float64_friendly());
    const float nnz_fraction = 0.05;
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const auto data =
        GENERATE_COPY(te::csr_table_builder(5, 5, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(7, 10, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(100, 100, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(1000, 1000, nnz_fraction, this->data_indexing_),
                      te::csr_table_builder(15000, 1000, nnz_fraction, this->data_indexing_));
    SKIP_IF(this->not_cpu_friendly(data));
    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));
    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);

    this->csr_general_checks(data, compute_mode);
}

} // namespace oneapi::dal::basic_statistics::test
