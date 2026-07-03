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

#include "oneapi/dal/algo/kmeans/test/fixture.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
namespace oneapi::dal::kmeans::test {

template <typename TestType>
class kmeans_batch_test : public kmeans_test<TestType, kmeans_batch_test<TestType>> {};

/*
TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans degenerated test",
                     "[kmeans][batch]",
                     kmeans_types) {
    // number of observations is equal to number of centroids (obvious clustering)
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    Float data[] = { 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 1.0 };
    const auto x = homogen_table::wrap(data, 3, 5);

    Float responses[] = { 0, 1, 2 };
    const auto y = homogen_table::wrap(responses, 3, 1);
    this->exact_checks(x, x, x, y, 3, 2, 0.0, 0.0, false);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test, "kmeans relocation test", "[kmeans][batch]", kmeans_types) {
    // relocation of empty cluster to the best candidate
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    Float data[] = { 0, 0, 0.5, 0, 0.5, 1, 1, 1 };
    const auto x = homogen_table::wrap(data, 4, 2);

    Float initial_centroids[] = { 0.5, 0.5, 3, 3 };
    const auto c_init = homogen_table::wrap(initial_centroids, 2, 2);

    Float final_centroids[] = { 0.25, 0, 0.75, 1 };
    const auto c_final = homogen_table::wrap(final_centroids, 2, 2);

    std::int64_t responses[] = { 0, 0, 1, 1 };
    const auto y = homogen_table::wrap(responses, 4, 1);

    Float expected_obj_function = 0.25;
    std::int64_t expected_n_iters = 4;
    this->exact_checks_with_reordering(x,
                                       c_init,
                                       c_final,
                                       y,
                                       2,
                                       expected_n_iters + 1,
                                       0.0,
                                       expected_obj_function,
                                       false);
}
*/

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans empty clusters test",
                     "[kmeans][batch]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());
    this->check_empty_clusters();
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans smoke train/infer test",
                     "[kmeans][batch]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());
    this->check_on_smoke_data();
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans train/infer on gold data",
                     "[kmeans][batch]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());
    this->check_on_gold_data();
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans block test",
                     "[kmeans][batch][nightly][block]",
                     kmeans_types) {
    // This test is not stable on CPU
    // TODO: Remove the following `SKIP_IF` once stability problem is resolved
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());
    this->check_on_large_data_with_one_cluster();
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans partial centroids stress test",
                     "[kmeans][batch][nightly][stress]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());
    this->partial_centroids_stress_test();
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "higgs: samples=1M, iters=3",
                     "[kmeans][batch][external-dataset][higgs]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    const std::int64_t iters = 3;
    const std::string higgs_path = "workloads/higgs/dataset/higgs_1m_test.csv";

    SECTION("clusters=10") {
        this->test_on_dataset(higgs_path, 10, iters, 3.1997724684, 14717484.0);
    }

    SECTION("clusters=100") {
        this->test_on_dataset(higgs_path, 100, iters, 2.7450205195, 10704352.0);
    }

    SECTION("cluster=250") {
        this->test_on_dataset(higgs_path, 250, iters, 2.5923397174, 9335216.0);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "susy: samples=0.5M, iters=10",
                     "[kmeans][nightly][batch][external-dataset][susy]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    const std::int64_t iters = 10;
    const std::string susy_path = "workloads/susy/dataset/susy_test.csv";

    SECTION("clusters=10") {
        this->test_on_dataset(susy_path, 10, iters, 1.7730860782, 3183696.0);
    }

    SECTION("clusters=100") {
        this->test_on_dataset(susy_path, 100, iters, 1.9384844916, 1757022.625);
    }

    SECTION("cluster=250") {
        this->test_on_dataset(susy_path, 250, iters, 1.8950113604, 1400958.5);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "epsilon: samples=80K, iters=2",
                     "[kmeans][nightly][batch][external-dataset][epsilon]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    const std::int64_t iters = 2;
    const std::string epsilon_path = "workloads/epsilon/dataset/epsilon_80k_train.csv";

    SECTION("clusters=512") {
        this->test_on_dataset(epsilon_path, 512, iters, 6.9367580565, 50128.640625, 1.0e-3);
    }

    SECTION("clusters=1024") {
        this->test_on_dataset(epsilon_path, 1024, iters, 5.59003873, 49518.75, 1.0e-3);
    }

    SECTION("cluster=2048") {
        this->test_on_dataset(epsilon_path, 2048, iters, 4.3202752143, 48437.6015625, 1.0e-3);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "higgs: samples=1M, iters=3 optional results",
                     "[kmeans][batch][external-dataset][higgs]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    const std::int64_t iters = 3;
    const std::string higgs_path = "workloads/higgs/dataset/higgs_1m_test.csv";

    SECTION("clusters=10") {
        this->test_optional_results_on_dataset(higgs_path, 10, iters, 3.1997724684, 14717484.0);
    }

    SECTION("clusters=100") {
        this->test_optional_results_on_dataset(higgs_path, 100, iters, 2.7450205195, 10704352.0);
    }

    SECTION("cluster=250") {
        this->test_optional_results_on_dataset(higgs_path, 250, iters, 2.5923397174, 9335216.0);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "susy: samples=0.5M, iters=10 optional results",
                     "[kmeans][nightly][batch][external-dataset][susy]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    const std::int64_t iters = 10;
    const std::string susy_path = "workloads/susy/dataset/susy_test.csv";

    SECTION("clusters=10") {
        this->test_optional_results_on_dataset(susy_path, 10, iters, 1.7730860782, 3183696.0);
    }

    SECTION("clusters=100") {
        this->test_optional_results_on_dataset(susy_path, 100, iters, 1.9384844916, 1757022.625);
    }

    SECTION("cluster=250") {
        this->test_optional_results_on_dataset(susy_path, 250, iters, 1.8950113604, 1400958.5);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "epsilon: samples=80K, iters=2 optional results",
                     "[kmeans][nightly][batch][external-dataset][epsilon]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    const std::int64_t iters = 2;
    const std::string epsilon_path = "workloads/epsilon/dataset/epsilon_80k_train.csv";

    SECTION("clusters=512") {
        this->test_optional_results_on_dataset(epsilon_path,
                                               512,
                                               iters,
                                               6.9367580565,
                                               50128.640625,
                                               1.0e-3);
    }

    SECTION("clusters=1024") {
        this->test_optional_results_on_dataset(epsilon_path,
                                               1024,
                                               iters,
                                               5.59003873,
                                               49518.75,
                                               1.0e-3);
    }

    SECTION("cluster=2048") {
        this->test_optional_results_on_dataset(epsilon_path,
                                               2048,
                                               iters,
                                               4.3202752143,
                                               48437.6015625,
                                               1.0e-3);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "KMmeans sparse default cases",
                     "[kmeans][batch][external-dataset]",
                     kmeans_types_csr) {
    SKIP_IF(!this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;

    const float nnz_fraction = 0.05;
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);

    SECTION("cluster=5") {
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(5,
                                                                      50,
                                                                      20,
                                                                      nnz_fraction,
                                                                      this->data_indexing_);
        bool init_centroids = true;
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=16") {
        bool init_centroids = true;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(16,
                                                                      200,
                                                                      100,
                                                                      nnz_fraction,
                                                                      this->data_indexing_);
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=128") {
        SKIP_IF(this->get_policy().is_cpu());
        bool init_centroids = true;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(128,
                                                                      100000,
                                                                      200,
                                                                      nnz_fraction,
                                                                      this->data_indexing_);
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=5") {
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(5,
                                                                      50,
                                                                      20,
                                                                      nnz_fraction,
                                                                      this->data_indexing_);
        bool init_centroids = false;
        this->test_on_sparse_data(input, 20, 0.01, init_centroids);
    }

    SECTION("cluster=16") {
        bool init_centroids = false;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(16,
                                                                      200,
                                                                      100,
                                                                      nnz_fraction,
                                                                      this->data_indexing_);
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=32") {
        SKIP_IF(this->get_policy().is_cpu());
        bool init_centroids = false;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(32,
                                                                      10000,
                                                                      100,
                                                                      nnz_fraction,
                                                                      this->data_indexing_);
        this->test_on_sparse_data(input, 30, 0.01, init_centroids);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "KMeans sparse repeated runs are deterministic",
                     "[kmeans][batch]",
                     kmeans_types_csr) {
    SKIP_IF(!this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_gpu());
    using Float = std::tuple_element_t<0, TestType>;

    // Guards against uninitialized reads in the centroid-shift path:
    // the same input + initial centroids must produce identical
    // centroids and objective value across independent train() calls.
    constexpr std::int64_t cluster_count = 16;
    constexpr std::int64_t row_count = 200;
    constexpr std::int64_t column_count = 100;
    constexpr std::int64_t max_iter = 10;
    constexpr float accuracy_threshold = 0.01;
    constexpr float nnz_fraction = 0.05;
    constexpr int trials = 5;

    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);

    auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(cluster_count,
                                                                  row_count,
                                                                  column_count,
                                                                  nnz_fraction,
                                                                  this->data_indexing_);
    const auto data = input.get_data(this->get_policy());
    const auto initial_centroids = input.get_initial_centroids();
    auto desc = this->get_descriptor(cluster_count, max_iter, accuracy_threshold);

    const auto baseline = this->train(desc, data, initial_centroids);
    const auto baseline_centroids =
        row_accessor<const Float>(baseline.get_model().get_centroids()).pull({ 0, -1 });
    const double baseline_obj = baseline.get_objective_function_value();
    const std::int64_t baseline_iters = baseline.get_iteration_count();

    for (int t = 1; t < trials; ++t) {
        const auto trial = this->train(desc, data, initial_centroids);
        REQUIRE(trial.get_iteration_count() == baseline_iters);
        REQUIRE(trial.get_objective_function_value() == baseline_obj);

        const auto trial_centroids =
            row_accessor<const Float>(trial.get_model().get_centroids()).pull({ 0, -1 });
        REQUIRE(trial_centroids.get_count() == baseline_centroids.get_count());
        for (std::int64_t i = 0; i < baseline_centroids.get_count(); ++i) {
            REQUIRE(trial_centroids[i] == baseline_centroids[i]);
        }
    }
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "KMeans dense and sparse agree on same input",
                     "[kmeans][batch]",
                     kmeans_types_csr) {
    SKIP_IF(!this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // csr_make_blobs generates data where ~nnz_fraction of entries per row are
    // stored; the remaining (1 - nnz_fraction) fraction is exactly zero. With
    // nnz_fraction = 0.05 we get ~95% zeros — a sizable amount of zeros, as
    // requested in review. The CSR and materialized dense views of the same
    // underlying arrays are the same input in two representations, so training
    // with identical initial centroids must give similar centroids, responses
    // and objective_function_value.
    constexpr std::int64_t cluster_count = 4;
    constexpr std::int64_t row_count = 100;
    constexpr std::int64_t column_count = 20;
    constexpr std::int64_t max_iter = 10;
    constexpr Float accuracy_threshold = Float(0);
    constexpr float nnz_fraction = 0.05f;

    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);

    const auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(cluster_count,
                                                                        row_count,
                                                                        column_count,
                                                                        nnz_fraction,
                                                                        this->data_indexing_);
    const auto csr_data = input.get_data(this->get_policy());
    const auto dense_data = input.get_dense_data(this->get_policy());
    const auto initial_centroids = input.get_initial_centroids();

    const auto csr_desc =
        kmeans::descriptor<Float, kmeans::method::lloyd_csr>{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iter)
            .set_accuracy_threshold(accuracy_threshold)
            .set_result_options(kmeans::result_options::compute_exact_objective_function);
    const auto dense_desc =
        kmeans::descriptor<Float, kmeans::method::lloyd_dense>{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iter)
            .set_accuracy_threshold(accuracy_threshold)
            .set_result_options(kmeans::result_options::compute_exact_objective_function);

    const auto csr_result = this->train(csr_desc, csr_data, initial_centroids);
    const auto dense_result = this->train(dense_desc, dense_data, initial_centroids);

    // Cluster labels are stable across methods because both start from the
    // same initial_centroids and update in-place — no reordering to worry
    // about, so a direct index-wise comparison is meaningful.
    const auto csr_centroids =
        row_accessor<const Float>(csr_result.get_model().get_centroids()).pull({ 0, -1 });
    const auto dense_centroids =
        row_accessor<const Float>(dense_result.get_model().get_centroids()).pull({ 0, -1 });
    REQUIRE(csr_centroids.get_count() == dense_centroids.get_count());
    const Float centroid_rel_tol = std::is_same_v<Float, double> ? Float(1e-8) : Float(1e-4);
    for (std::int64_t i = 0; i < csr_centroids.get_count(); ++i) {
        REQUIRE(
            this->check_value_with_ref_tol(csr_centroids[i], dense_centroids[i], centroid_rel_tol));
    }

    this->check_response_match(csr_result.get_responses(), dense_result.get_responses());

    // Dense and CSR use different summation orderings for the objective
    // (BLAS-gemm-based vs per-nnz), so the objective agreement is limited
    // by float rounding: ~1e-4 in single precision.
    const Float csr_obj = static_cast<Float>(csr_result.get_objective_function_value());
    const Float dense_obj = static_cast<Float>(dense_result.get_objective_function_value());
    const Float obj_rel_tol = std::is_same_v<Float, double> ? Float(1e-10) : Float(1e-3);
    CAPTURE(csr_obj, dense_obj);
    REQUIRE(this->check_value_with_ref_tol(csr_obj, dense_obj, obj_rel_tol));
}

#ifdef ONEDAL_DATA_PARALLEL

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "KMmeans sparse cases on large number of rows",
                     "[kmeans][batch][external-dataset]",
                     kmeans_types_csr) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(!this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Check that algorithm does not crash on big number of rows
    constexpr std::int64_t cluster_count = 5;
    std::int64_t rows_count = 1000 * 1000;
    auto device = this->get_queue().get_device();
    std::string device_name = device.template get_info<sycl::info::device::name>();
    if (device_name.find("Data Center GPU Max") != std::string::npos) {
        rows_count = 100 * 1000 * 1000;
    }
    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(cluster_count,
                                                                  rows_count,
                                                                  20,
                                                                  0.05,
                                                                  this->data_indexing_);

    auto desc = this->get_descriptor(cluster_count, 10, 0.01);
    const table initial_centroids = input.get_initial_centroids();
    const table data = input.get_data(this->get_policy());
    const auto train_result = this->train(desc, data, initial_centroids);
}

#endif

} // namespace oneapi::dal::kmeans::test
