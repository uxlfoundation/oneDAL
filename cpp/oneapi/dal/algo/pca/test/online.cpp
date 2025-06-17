/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/pca/test/fixture.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace pca = oneapi::dal::pca;
using pca_types_cov = COMBINE_TYPES((float, double), (pca::method::cov));
using pca_types_svd = COMBINE_TYPES((float, double), (pca::method::svd));

template <typename TestType>
class pca_online_test : public pca_test<TestType, pca_online_test<TestType>> {};

TEMPLATE_LIST_TEST_M(pca_online_test,
                     "pca online on gold data",
                     "[pca][integration][online][gold]",
                     pca_types_cov) {
    SKIP_IF(this->not_float64_friendly());
    const int64_t nBlocks = GENERATE(1, 3, 10);

    const std::int64_t component_count = 0;
    const bool deterministic = true;
    const auto pca_desc = this->get_descriptor(component_count, deterministic);
    const auto gold_data = this->get_gold_data();

    auto partial_result = dal::pca::partial_train_result();
    auto input_table = this->template split_table_by_rows<float>(gold_data, nBlocks);
    for (std::int64_t i = 0; i < nBlocks; ++i) {
        partial_result = this->partial_train(pca_desc, partial_result, input_table[i]);
    }

    const auto pca_result = this->finalize_train(pca_desc, partial_result);

    const auto eigenvalues = pca_result.get_eigenvalues();
    const auto eigenvectors = pca_result.get_eigenvectors();

    SECTION("check eigenvalues") {
        const auto gold_eigenvalues = this->get_gold_eigenvalues();
        this->check_eigenvalues(gold_eigenvalues, eigenvalues);
    }

    SECTION("check eigenvectors") {
        const auto gold_eigenvectors = this->get_gold_eigenvectors();
        this->check_eigenvectors(gold_eigenvectors, eigenvectors);
    }
}

TEMPLATE_LIST_TEST_M(pca_online_test,
                     "pca fill_uniform flow cov",
                     "[pca][integration][online]",
                     pca_types_cov) {
    SKIP_IF(this->not_float64_friendly());

    const int64_t nBlocks = GENERATE(1, 3, 10);
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 100, 100 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 1000, 100 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 10000, 100 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 100000, 10 }.fill_uniform(-0.2, 1.5));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto data_table_id = this->get_homogen_table_id();

    const std::int64_t component_count = GENERATE_COPY(0,
                                                       1,
                                                       data.get_column_count(),
                                                       data.get_column_count() - 1,
                                                       data.get_column_count() / 2);

    this->online_general_checks(data, component_count, data_table_id, nBlocks);
}

TEMPLATE_LIST_TEST_M(pca_online_test,
                     "pca fill_normal flow svd",
                     "[pca][integration][online]",
                     pca_types_svd) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->not_available_on_device());

    const int64_t nBlocks = GENERATE(1, 3, 10);
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 1000, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 10000, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100000, 10 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto data_table_id = this->get_homogen_table_id();

    const std::int64_t component_count = GENERATE_COPY(0,
                                                       1,
                                                       data.get_column_count(),
                                                       data.get_column_count() - 1,
                                                       data.get_column_count() / 2);

    this->online_general_checks(data, component_count, data_table_id, nBlocks);
}

TEMPLATE_LIST_TEST_M(pca_online_test,
                     "pca fill_normal flow",
                     "[pca][integration][online]",
                     pca_types_cov) {
    SKIP_IF(this->not_float64_friendly());

    const int64_t nBlocks = GENERATE(1, 3, 10);
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 1000, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 10000, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100000, 10 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto data_table_id = this->get_homogen_table_id();

    const std::int64_t component_count = GENERATE_COPY(0,
                                                       1,
                                                       data.get_column_count(),
                                                       data.get_column_count() - 1,
                                                       data.get_column_count() / 2);

    this->online_general_checks(data, component_count, data_table_id, nBlocks);
}

TEMPLATE_LIST_TEST_M(pca_online_test,
                     "pca common flow higgs",
                     "[external-dataset][pca][integration][online]",
                     pca_types_cov) {
    SKIP_IF(this->not_float64_friendly());

    const int64_t nBlocks = GENERATE(1, 3, 10);
    const std::int64_t component_count = 1;
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_100t_train.csv" });

    const auto data_table_id = this->get_homogen_table_id();

    this->online_general_checks(data, component_count, data_table_id, nBlocks);
}

} // namespace oneapi::dal::pca::test
