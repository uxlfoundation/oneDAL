/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/kmeans/test/data.hpp"
#include "oneapi/dal/algo/kmeans/test/spmd_backend_fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/memory_impl_dpc.hpp"

namespace oneapi::dal::kmeans::test {

#ifdef ONEDAL_DATA_PARALLEL

template <typename TestType>
class kmeans_spmd_backend_test
        : public kmeans_spmd_backend_fixture<PARALLEL_BACKEND,
                                             TestType,
                                             kmeans_spmd_backend_test<TestType>> {
public:
    using base_t =
        kmeans_spmd_backend_fixture<PARALLEL_BACKEND, TestType, kmeans_spmd_backend_test<TestType>>;
    using float_t = typename base_t::float_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

    template <typename... Args>
    train_result_t train_override(Args&&... args) {
        return this->train_in_parallel_and_merge(std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<train_input_t> split_train_input_override(std::int64_t split_count,
                                                          Args&&... args) {
        // Data table is distributed across the ranks, but
        // initial centroids are common for all the ranks
        const train_input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);
        const auto common_centroids = input.get_initial_centroids();

        std::vector<train_input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back( //
                train_input_t{ split_data[i] } //
                    .set_initial_centroids(common_centroids) //
            );
        }

        return split_input;
    }

    train_result_t merge_train_result_override(const train_result_t& local_result) {
        auto local_responses = local_result.get_responses();
        auto row_count = local_responses.get_row_count();
        ONEDAL_ASSERT(local_responses.get_column_count() == 1);

        auto arr_temp = row_accessor<const float_t>(local_responses)
                            .pull(this->get_queue(), { 0, -1 }, sycl::usm::alloc::device);
        auto arr_local = oneapi::dal::array<float_t>::empty(this->get_queue(),
                                                            row_count,
                                                            sycl::usm::alloc::device);
        dal::detail::memcpy(this->get_queue(),
                            arr_local.get_mutable_data(),
                            arr_temp.get_data(),
                            sizeof(float_t) * row_count);

        std::int64_t total_row_count = row_count;
        auto comm = this->get_comm();
        std::int64_t rank = comm.get_rank();
        std::int64_t rank_count = comm.get_rank_count();
        comm.allreduce(total_row_count).wait();
        auto arr_total = oneapi::dal::array<float_t>::empty(this->get_queue(),
                                                            total_row_count,
                                                            sycl::usm::alloc::device);
        auto recv_counts = dal::array<std::int64_t>::zeros(rank_count);
        recv_counts.get_mutable_data()[rank] = row_count;
        comm.allreduce(recv_counts, spmd::reduce_op::sum).wait();
        auto displs = array<std::int64_t>::zeros(rank_count);
        auto displs_ptr = displs.get_mutable_data();
        std::int64_t total_count = 0;
        for (std::int64_t i = 0; i < rank_count; i++) {
            displs_ptr[i] = total_count;
            total_count += recv_counts.get_data()[i];
        }
        comm.allgatherv(arr_local, arr_total, recv_counts.get_data(), displs.get_data()).wait();

        return train_result_t{}
            .set_responses(dal::homogen_table::wrap(arr_total, total_row_count, 1))
            .set_iteration_count(local_result.get_iteration_count())
            .set_objective_function_value(local_result.get_objective_function_value())
            .set_model(local_result.get_model());
    }

    void check_if_results_same_on_all_ranks() {
        const auto table_id = this->get_homogen_table_id();
        const auto data = gold_dataset::get_data().get_table(table_id);
        const auto initial_centroids = gold_dataset::get_initial_centroids().get_table(table_id);

        const std::int64_t cluster_count = gold_dataset::get_cluster_count();
        constexpr std::int64_t max_iteration_count = 100;
        constexpr float_t accuracy_threshold = 0.0;

        INFO("create descriptor");
        const auto kmeans_desc =
            this->get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto result = this->train_in_parallel(kmeans_desc, data, initial_centroids);

        SECTION("check if all results bitwise equal on all ranks") {
            SECTION("check centroids") {
                auto comm = this->get_comm();
                const auto centroids = result.get_model().get_centroids();
                const std::int64_t cluster_count = centroids.get_row_count();
                const std::int64_t column_count = centroids.get_column_count();
                std::int64_t min_value = cluster_count;
                comm.allreduce(min_value, dal::preview::spmd::reduce_op::min).wait();
                std::int64_t max_value = cluster_count;
                comm.allreduce(max_value, dal::preview::spmd::reduce_op::max).wait();
                REQUIRE(min_value == max_value);

                auto arr_local = oneapi::dal::array<float_t>::zeros(this->get_queue(),
                                                                    cluster_count * column_count,
                                                                    sycl::usm::alloc::device);
                constexpr std::int64_t root_rank = 0;
                const std::int64_t rank = comm.get_rank();
                if (rank == root_rank) {
                    auto arr_temp =
                        row_accessor<const float_t>(centroids).pull(this->get_queue(),
                                                                    { 0, -1 },
                                                                    sycl::usm::alloc::device);

                    dal::detail::memcpy(this->get_queue(),
                                        arr_local.get_mutable_data(),
                                        arr_temp.get_data(),
                                        sizeof(float_t) * cluster_count * column_count);
                }
                comm.bcast(arr_local).wait();
                if (rank != root_rank) {
                    auto front_centroids =
                        dal::homogen_table::wrap(arr_local, cluster_count, column_count);
                    te::check_if_tables_equal<float_t>(centroids, front_centroids);
                }
            }

            SECTION("check iterations") {
                auto comm = this->get_comm();
                std::int64_t min_value = result.get_iteration_count();
                comm.allreduce(min_value, dal::preview::spmd::reduce_op::min).wait();
                std::int64_t max_value = result.get_iteration_count();
                comm.allreduce(max_value, dal::preview::spmd::reduce_op::max).wait();
                REQUIRE(min_value == max_value);
            }

            SECTION("check objective function") {
                auto comm = this->get_comm();
                float_t min_value = result.get_objective_function_value();
                comm.allreduce(min_value, dal::preview::spmd::reduce_op::min).wait();
                float_t max_value = result.get_objective_function_value();
                comm.allreduce(max_value, dal::preview::spmd::reduce_op::max).wait();
                REQUIRE(min_value == max_value);
            }
        }
    }
};

TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "make sure results are the same on all ranks",
                     "[spmd][smoke]",
                     kmeans_types) {
    // SPMD mode is not implemented for CPU. The following `SKIP_IF` should be
    // removed once it's supported for CPU. The same for the rest of tests cases.
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->check_if_results_same_on_all_ranks();
}
TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "distributed kmeans empty clusters test",
                     "[spmd][smoke]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->check_empty_clusters();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "distributed kmeans smoke train/infer test",
                     "[spmd][smoke]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->check_on_smoke_data();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "distributed kmeans train/infer on gold data",
                     "[spmd][smoke]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->check_on_gold_data();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "distributed kmeans block test",
                     "[spmd][block][nightly]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->check_on_large_data_with_one_cluster();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "distributed higgs: samples=1M, iters=3",
                     "[kmeans][spmd][higgs][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

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

TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "distributed susy: samples=0.5M, iters=10",
                     "[kmeans][nightly][spmd][susy][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

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

TEMPLATE_LIST_TEST_M(kmeans_spmd_backend_test,
                     "distributed epsilon: samples=80K, iters=2",
                     "[kmeans][nightly][spmd][epsilon][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t iters = 2;
    const std::string epsilon_path = "workloads/epsilon/dataset/epsilon_80k_train.csv";

    SECTION("clusters=512") {
        this->test_on_dataset(epsilon_path, 512, iters, 6.9367580565, 50128.640625, 1.0e-3);
    }
}

#endif
} // namespace oneapi::dal::kmeans::test
