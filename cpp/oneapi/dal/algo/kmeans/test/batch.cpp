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
                     "[kmeans][batch]",
                     kmeans_types) {
    SKIP_IF(!this->is_sparse_method());
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;

    SECTION("cluster=5") {
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(5, 50, 20);
        bool init_centroids = true;
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=16") {
        bool init_centroids = true;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(16, 200, 100);
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=128") {
        SKIP_IF(this->get_policy().is_cpu());
        bool init_centroids = true;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(128, 100000, 200);
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=5") {
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(5, 50, 20);
        bool init_centroids = false;
        this->test_on_sparse_data(input, 20, 0.01, init_centroids);
    }

    SECTION("cluster=16") {
        bool init_centroids = false;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(16, 200, 100);
        this->test_on_sparse_data(input, 10, 0.01, init_centroids);
    }

    SECTION("cluster=32") {
        SKIP_IF(this->get_policy().is_cpu());
        bool init_centroids = false;
        auto input = oneapi::dal::test::engine::csr_make_blobs<Float>(32, 10000, 100);
        this->test_on_sparse_data(input, 30, 0.01, init_centroids);
    }
}

} // namespace oneapi::dal::kmeans::test
