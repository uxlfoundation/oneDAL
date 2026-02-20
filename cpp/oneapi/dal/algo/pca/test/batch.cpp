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
using pca_types = COMBINE_TYPES((float, double), (pca::method::cov, pca::method::svd));
using pca_types_cov = COMBINE_TYPES((float, double), (pca::method::cov));
using pca_types_svd = COMBINE_TYPES((float, double), (pca::method::svd));
using pca_types_precomputed = COMBINE_TYPES((float, double), (method::precomputed));

template <typename TestType>
class pca_batch_test : public pca_test<TestType, pca_batch_test<TestType>> {};

TEMPLATE_LIST_TEST_M(pca_batch_test, "pca common flow", "[pca][integration][batch]", pca_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 100000, 10 }.fill_uniform(-0.2, 1.5));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto data_table_id = this->get_homogen_table_id();

    const std::int64_t component_count = GENERATE_COPY(0,
                                                       1,
                                                       data.get_column_count(),
                                                       data.get_column_count() - 1,
                                                       data.get_column_count() / 2);

    this->general_checks(data, component_count, data_table_id);
}

TEMPLATE_LIST_TEST_M(pca_batch_test,
                     "pca on gold data",
                     "[pca][integration][batch][gold]",
                     pca_types) {
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t component_count = 0;
    const bool deterministic = true;
    const auto pca_desc = this->get_descriptor(component_count, deterministic);
    const auto gold_data = this->get_gold_data();

    const auto pca_result = te::train(this->get_policy(), pca_desc, gold_data);
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

TEMPLATE_LIST_TEST_M(pca_batch_test,
                     "pca common flow higgs",
                     "[external-dataset][pca][integration][batch]",
                     pca_types) {
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t component_count = 0;
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_100t_train.csv" });

    const auto data_table_id = this->get_homogen_table_id();

    this->general_checks(data, component_count, data_table_id);
}

TEMPLATE_LIST_TEST_M(pca_batch_test,
                     "pca with cov",
                     "[pca][integration][precomputed][batch]",
                     pca_types_precomputed) {
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t component_count = 0;
    const bool deterministic = true;
    const auto pca_desc = this->get_descriptor(component_count, deterministic);
    const auto gold_data = this->get_gold_cor();

    const auto pca_result = te::train(this->get_policy(), pca_desc, gold_data);
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

} // namespace oneapi::dal::pca::test
