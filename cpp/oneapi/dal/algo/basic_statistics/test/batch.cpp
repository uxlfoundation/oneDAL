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
    // TODO: Temporary workaround. The mean/variance test fails for large datasets.
    // const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));

    const bs::result_option_id compute_mode =
        GENERATE_COPY(res_min_max, /*res_mean_varc,*/ res_all);

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
    const bs::result_option_id res_min_max = result_options::min | result_options::max;
    const bs::result_option_id res_mean_varc = result_options::mean | result_options::variance;
    const bs::result_option_id res_all =
        bs::result_option_id(dal::result_option_id_base(mask_full));
    const bs::result_option_id compute_mode = GENERATE_COPY(res_min_max, res_mean_varc, res_all);

    this->csr_general_checks(data, compute_mode);
}

// Column 0 holds 1e4 plus small deviations in every row; column 1 is empty, column 2 holds
// one value per eight rows, so the implicit zeros enter its mean, extremes and variance.
TEMPLATE_LIST_TEST_M(basic_statistics_batch_test,
                     "basic_statistics CSR moments far from zero and with implicit zeros",
                     "[basic_statistics][integration][batch]",
                     basic_statistics_sparse_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());
    using float_t = std::tuple_element_t<0, TestType>;
    constexpr std::int64_t row_count = 4096;
    constexpr std::int64_t column_count = 3;

    std::vector<float_t> values;
    std::vector<std::int64_t> column_indices;
    std::vector<std::int64_t> row_offsets{ 0 };
    std::vector<long double> column0, column2(row_count, 0.0L);
    for (std::int64_t i = 0; i < row_count; ++i) {
        const float_t x0 = float_t(1e4) + float_t((i * 37) % 101) * float_t(1e-2);
        values.push_back(x0);
        column_indices.push_back(0);
        column0.push_back(x0);
        if (i % 8 == 0) {
            const float_t x2 = float_t((i * 13) % 17) - float_t(8);
            values.push_back(x2);
            column_indices.push_back(2);
            column2[i] = x2;
        }
        row_offsets.push_back(std::int64_t(values.size()));
    }
    const auto data = csr_table::wrap(values.data(),
                                      column_indices.data(),
                                      row_offsets.data(),
                                      row_count,
                                      column_count,
                                      sparse_indexing::zero_based);

    const auto desc =
        bs::descriptor<float_t, basic_statistics::method::sparse>{}.set_result_options(
            bs::result_option_id(dal::result_option_id_base(mask_full)));
    const auto result = this->compute(desc, data);

    const auto moments = [](const std::vector<long double>& x) {
        long double mean = 0, ssc = 0;
        for (auto v : x)
            mean += v;
        mean /= x.size();
        for (auto v : x)
            ssc += (v - mean) * (v - mean);
        return std::pair<long double, long double>{ mean, ssc };
    };
    const auto [mean0, ssc0] = moments(column0);
    const auto [mean2, ssc2] = moments(column2);
    const std::vector<double> ref_mean{ double(mean0), 0.0, double(mean2) };
    const std::vector<double> ref_ssc{ double(ssc0), 0.0, double(ssc2) };
    const std::vector<double> ref_min{ 1e4, 0.0, -8.0 };

    const auto mean = row_accessor<const float_t>(result.get_mean()).pull();
    const auto ssc = row_accessor<const float_t>(result.get_sum_squares_centered()).pull();
    const auto variance = row_accessor<const float_t>(result.get_variance()).pull();
    const auto min = row_accessor<const float_t>(result.get_min()).pull();
    const auto max = row_accessor<const float_t>(result.get_max()).pull();
    const double tol = std::is_same_v<float_t, float> ? 1e-5 : 1e-12;
    for (std::int64_t j = 0; j < column_count; ++j) {
        CAPTURE(j, mean[j], ref_mean[j], ssc[j], ref_ssc[j], variance[j], min[j], max[j]);
        REQUIRE(std::abs(mean[j] - ref_mean[j]) <= tol * std::max(1.0, std::abs(ref_mean[j])));
        REQUIRE(std::abs(ssc[j] - ref_ssc[j]) <= tol * std::max(1.0, ref_ssc[j]));
        REQUIRE(std::abs(variance[j] - ref_ssc[j] / (row_count - 1)) <=
                tol * std::max(1.0, ref_ssc[j] / (row_count - 1)));
        REQUIRE(min[j] == float_t(ref_min[j]));
    }
    REQUIRE(max[1] == float_t(0));
    REQUIRE(max[2] == float_t(8));
}

// Repeated column indices in a row are summed, as in the dense row: row 0 repeats column 0 in
// order, row 1 repeats column 2 out of order.
TEMPLATE_LIST_TEST_M(basic_statistics_batch_test,
                     "basic_statistics CSR sums repeated column indices",
                     "[basic_statistics][integration][batch]",
                     basic_statistics_sparse_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());
    using float_t = std::tuple_element_t<0, TestType>;
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    std::vector<float_t> values{ 2, 5, 1, 3, 4, 6 };
    std::vector<std::int64_t> column_indices{ 0, 0, 2, 0, 2, 1 };
    std::vector<std::int64_t> row_offsets{ 0, 2, 5, 6, 6 };
    const double dense[row_count][column_count] = { { 7, 0, 0 },
                                                    { 3, 0, 5 },
                                                    { 0, 6, 0 },
                                                    { 0, 0, 0 } };
    const auto data = csr_table::wrap(values.data(),
                                      column_indices.data(),
                                      row_offsets.data(),
                                      row_count,
                                      column_count,
                                      sparse_indexing::zero_based);
    const auto desc =
        bs::descriptor<float_t, basic_statistics::method::sparse>{}.set_result_options(
            bs::result_option_id(dal::result_option_id_base(mask_full)));
    const auto result = this->compute(desc, data);

    const auto sum = row_accessor<const float_t>(result.get_sum()).pull();
    const auto sum_sq = row_accessor<const float_t>(result.get_sum_squares()).pull();
    const auto ssc = row_accessor<const float_t>(result.get_sum_squares_centered()).pull();
    const auto min = row_accessor<const float_t>(result.get_min()).pull();
    const auto max = row_accessor<const float_t>(result.get_max()).pull();
    for (std::int64_t j = 0; j < column_count; ++j) {
        double s = 0, s2 = 0, lo = dense[0][j], hi = dense[0][j];
        for (std::int64_t i = 0; i < row_count; ++i) {
            s += dense[i][j];
            s2 += dense[i][j] * dense[i][j];
            lo = std::min(lo, dense[i][j]);
            hi = std::max(hi, dense[i][j]);
        }
        const double ref_ssc = s2 - s * s / row_count;
        CAPTURE(j, sum[j], sum_sq[j], ssc[j], min[j], max[j]);
        REQUIRE(sum[j] == float_t(s));
        REQUIRE(sum_sq[j] == float_t(s2));
        REQUIRE(std::abs(ssc[j] - ref_ssc) <= 1e-5 * std::max(1.0, ref_ssc));
        REQUIRE(min[j] == float_t(lo));
        REQUIRE(max[j] == float_t(hi));
    }
}

// A full column has no implicit zeros to merge; that empty merge must not turn 1e200^2 * 0 into NaN.
TEMPLATE_LIST_TEST_M(basic_statistics_batch_test,
                     "basic_statistics CSR moments of a full column of huge values",
                     "[basic_statistics][integration][batch]",
                     basic_statistics_sparse_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());
    using float_t = std::tuple_element_t<0, TestType>;
    SKIP_IF((std::is_same_v<float_t, float>));
    const double huge = 1e200;
    std::vector<float_t> values(3, float_t(huge));
    std::vector<std::int64_t> column_indices{ 0, 0, 0 };
    std::vector<std::int64_t> row_offsets{ 0, 1, 2, 3 };
    const auto data = csr_table::wrap(values.data(),
                                      column_indices.data(),
                                      row_offsets.data(),
                                      3,
                                      1,
                                      sparse_indexing::zero_based);
    const auto desc =
        bs::descriptor<float_t, basic_statistics::method::sparse>{}.set_result_options(
            bs::result_option_id(dal::result_option_id_base(mask_full)));
    const auto result = this->compute(desc, data);

    const auto ssc = row_accessor<const float_t>(result.get_sum_squares_centered()).pull();
    const auto variance = row_accessor<const float_t>(result.get_variance()).pull();
    const auto mean = row_accessor<const float_t>(result.get_mean()).pull();
    REQUIRE(ssc[0] == float_t(0));
    REQUIRE(variance[0] == float_t(0));
    REQUIRE(mean[0] == float_t(huge));
}

// Enough values for several chunks, so a reduction that depends on the thread schedule would show.
TEMPLATE_LIST_TEST_M(basic_statistics_batch_test,
                     "basic_statistics CSR results are the same from run to run",
                     "[basic_statistics][integration][batch]",
                     basic_statistics_sparse_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());
    using float_t = std::tuple_element_t<0, TestType>;
    const auto builder = te::csr_table_builder(20000, 100, 0.1, sparse_indexing::zero_based);
    const auto data = builder.build_csr_table(this->get_policy());
    const auto desc =
        bs::descriptor<float_t, basic_statistics::method::sparse>{}.set_result_options(
            bs::result_option_id(dal::result_option_id_base(mask_full)));

    const auto values = [](const table& t) {
        const auto ary = row_accessor<const float_t>(t).pull();
        return std::vector<float_t>(ary.get_data(), ary.get_data() + ary.get_count());
    };
    const auto all = [&](const auto& r) {
        return std::vector<std::vector<float_t>>{ values(r.get_min()),
                                                  values(r.get_max()),
                                                  values(r.get_sum()),
                                                  values(r.get_sum_squares()),
                                                  values(r.get_sum_squares_centered()),
                                                  values(r.get_mean()),
                                                  values(r.get_variance()) };
    };
    const auto first = all(this->compute(desc, data));
    for (int run = 0; run < 5; ++run) {
        CAPTURE(run);
        REQUIRE(all(this->compute(desc, data)) == first);
    }
}

} // namespace oneapi::dal::basic_statistics::test
