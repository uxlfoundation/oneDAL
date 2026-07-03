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

#include <algorithm>
#include <numeric>
#include <random>

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
                     "KMeans CSR exact objective accounts for unstored features",
                     "[kmeans][batch]",
                     kmeans_types_csr) {
    SKIP_IF(!this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Regression test for the CSR exact-objective bug. Before the fix,
    // PostProcessing<lloydCSR>::computeExactObjectiveFunction summed
    // (x_j - c_j)^2 only over the stored entries of a row, which drops the
    // c_j^2 contribution from features where the row has an implicit zero
    // and the centroid is non-zero. The previous csr_make_blobs-based test
    // did not trigger this: rows there share the column support of their
    // centroid, so c_j = 0 at every unstored index and the missing term is
    // identically zero -- the buggy path returned the same value as dense.
    //
    // No random seed is needed: initial_centroids are passed explicitly to
    // train(), so the k-means init algorithm is skipped and the Lloyd loop
    // runs deterministically from a fixed starting point.
    constexpr std::int64_t row_count = 3;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t cluster_count = 2;
    constexpr std::int64_t max_iter = 1;
    constexpr double accuracy_threshold = 0.0;

    const Float data_vals[] = { Float(10), Float(10), Float(20) };
    const Float dense_vals[] = { Float(10), Float(0), Float(0), Float(0), Float(10),
                                 Float(0),  Float(0), Float(0), Float(20) };
    const Float init_centroids_vals[] = { Float(5), Float(5), Float(0),
                                          Float(0), Float(0), Float(20) };

    const auto indexing = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    const std::int64_t shift = (indexing == sparse_indexing::one_based) ? 1 : 0;
    const std::int64_t col_indices[] = { 0 + shift, 1 + shift, 2 + shift };
    const std::int64_t row_offsets[] = { 0 + shift, 1 + shift, 2 + shift, 3 + shift };

    const auto csr_data = csr_table::wrap<Float>(data_vals,
                                                 col_indices,
                                                 row_offsets,
                                                 row_count,
                                                 column_count,
                                                 indexing);
    const auto dense_data = homogen_table::wrap(dense_vals, row_count, column_count);
    const auto initial_centroids =
        homogen_table::wrap(init_centroids_vals, cluster_count, column_count);

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

    // Both start from the same explicit initial centroids and update
    // in-place, so responses correspond index-wise and centroids match
    // numerically.
    this->check_response_match(csr_result.get_responses(), dense_result.get_responses());

    const auto csr_centroids =
        row_accessor<const Float>(csr_result.get_model().get_centroids()).pull({ 0, -1 });
    const auto dense_centroids =
        row_accessor<const Float>(dense_result.get_model().get_centroids()).pull({ 0, -1 });
    REQUIRE(csr_centroids.get_count() == dense_centroids.get_count());
    const Float centroid_tol = std::is_same_v<Float, double> ? Float(1e-10) : Float(1e-5);
    for (std::int64_t i = 0; i < csr_centroids.get_count(); ++i) {
        REQUIRE(this->check_value_with_ref_tol(csr_centroids[i], dense_centroids[i], centroid_tol));
    }

    // Core assertion: dense objective is 100, buggy CSR objective is 50;
    // this fails on main and passes with the postprocessing fix.
    const Float csr_obj = static_cast<Float>(csr_result.get_objective_function_value());
    const Float dense_obj = static_cast<Float>(dense_result.get_objective_function_value());
    const Float obj_tol = std::is_same_v<Float, double> ? Float(1e-10) : Float(1e-5);
    CAPTURE(csr_obj, dense_obj);
    REQUIRE(this->check_value_with_ref_tol(csr_obj, dense_obj, obj_tol));
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "KMeans CSR run with a seeded random initialization is reproducible",
                     "[kmeans][batch]",
                     kmeans_types_csr) {
    SKIP_IF(!this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    // Complements the previous case: instead of hand-picked initial
    // centroids, pick them by drawing `cluster_count` distinct row indices
    // from the sparse dataset with a fixed seed. Since the seed is fixed,
    // two independent CSR runs must be bit-identical, and the CSR run of
    // the same shuffle must recover the same cluster centers as the dense
    // run (up to label permutation and fp summation ordering).
    //
    // The response comparison across the CSR and dense paths is relaxed
    // rather than strict: the distance math is not fp-equivalent between
    // the two representations (dense sums (x_j - c_j)^2 over all features,
    // CSR sums (x_j^2 - 2 x_j c_j) over stored features and adds ||c||^2),
    // so a point near a decision boundary can flip cluster on one path
    // only due to rounding. Over `max_iter` iterations these differences
    // can amplify, so we allow a small fraction of points to disagree on
    // cluster assignment while still requiring centroid agreement within
    // tolerance and objective agreement within tolerance.
    constexpr std::int64_t cluster_count = 4;
    constexpr std::int64_t row_count = 100;
    constexpr std::int64_t column_count = 20;
    constexpr std::int64_t max_iter = 10;
    constexpr float nnz_fraction = 0.05f;
    constexpr Float accuracy_threshold = Float(0);
    constexpr std::uint32_t seed = 777u;

    this->data_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);

    const auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(cluster_count,
                                                                        row_count,
                                                                        column_count,
                                                                        nnz_fraction,
                                                                        this->data_indexing_);
    const auto csr_data = input.get_data(this->get_policy());
    const auto dense_data = input.get_dense_data(this->get_policy());

    // Build initial centroids by seeded row sampling from the dense view.
    const auto pick_seeded_initial_centroids = [&](std::uint32_t s) {
        const auto dense_rows = row_accessor<const Float>(dense_data).pull({ 0, -1 });
        std::vector<std::int64_t> indices(row_count);
        std::iota(indices.begin(), indices.end(), std::int64_t(0));
        std::mt19937 rng(s);
        std::shuffle(indices.begin(), indices.end(), rng);
        auto centroids = array<Float>::empty(cluster_count * column_count);
        auto* const out = centroids.get_mutable_data();
        for (std::int64_t k = 0; k < cluster_count; ++k) {
            const std::int64_t src = indices[k];
            for (std::int64_t j = 0; j < column_count; ++j) {
                out[k * column_count + j] = dense_rows[src * column_count + j];
            }
        }
        return homogen_table::wrap(centroids, cluster_count, column_count);
    };

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

    const auto init1 = pick_seeded_initial_centroids(seed);
    const auto init2 = pick_seeded_initial_centroids(seed);

    const auto csr_run_a = this->train(csr_desc, csr_data, init1);
    const auto csr_run_b = this->train(csr_desc, csr_data, init2);
    const auto dense_run = this->train(dense_desc, dense_data, init1);

    // Same seed on the sparse path -> bit-identical objective and iteration
    // count across independent runs.
    REQUIRE(csr_run_a.get_iteration_count() == csr_run_b.get_iteration_count());
    REQUIRE(csr_run_a.get_objective_function_value() == csr_run_b.get_objective_function_value());

    // Same seed shared between sparse and dense representations -> centroids
    // must agree within tolerance after matching by nearest centroid, and
    // per-point cluster assignment must agree for the vast majority of rows
    // (a small fraction can flip near decision boundaries due to fp).
    const auto csr_centroids = csr_run_a.get_model().get_centroids();
    const auto dense_centroids = dense_run.get_model().get_centroids();
    auto match_map = array<Float>::zeros(cluster_count);
    this->find_match_centroids(dense_centroids, csr_centroids, column_count, match_map);
    const Float centroid_rel_tol = std::is_same_v<Float, double> ? Float(1e-6) : Float(1e-2);
    this->check_centroid_match_with_rel_tol(match_map,
                                            centroid_rel_tol,
                                            dense_centroids,
                                            csr_centroids);

    const auto csr_responses = row_accessor<const Float>(csr_run_a.get_responses()).pull({ 0, -1 });
    const auto dense_responses =
        row_accessor<const Float>(dense_run.get_responses()).pull({ 0, -1 });
    REQUIRE(csr_responses.get_count() == dense_responses.get_count());
    std::int64_t mismatch_count = 0;
    for (std::int64_t i = 0; i < csr_responses.get_count(); ++i) {
        // find_match_centroids(dense, csr, ...) sets match_map[csr_label]
        // to the corresponding label in dense-centroid space.
        if (dense_responses[i] != match_map[static_cast<std::int64_t>(csr_responses[i])]) {
            ++mismatch_count;
        }
    }
    const double mismatch_ratio =
        static_cast<double>(mismatch_count) / static_cast<double>(csr_responses.get_count());
    const double mismatch_tol = 0.05;
    CAPTURE(mismatch_count, csr_responses.get_count(), mismatch_ratio, mismatch_tol);
    REQUIRE(mismatch_ratio <= mismatch_tol);

    const Float csr_obj = static_cast<Float>(csr_run_a.get_objective_function_value());
    const Float dense_obj = static_cast<Float>(dense_run.get_objective_function_value());
    const Float obj_rel_tol = std::is_same_v<Float, double> ? Float(1e-6) : Float(1e-2);
    CAPTURE(seed, csr_obj, dense_obj);
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
