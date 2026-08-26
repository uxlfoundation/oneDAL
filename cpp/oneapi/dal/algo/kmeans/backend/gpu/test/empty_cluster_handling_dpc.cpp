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

#include <random>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/thread_communicator.hpp"

#include "oneapi/dal/algo/kmeans/backend/gpu/empty_cluster_handling.hpp"

namespace oneapi::dal::kmeans::backend::test {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace spmd = dal::preview::spmd;
namespace te = dal::test::engine;

// TODO: Move to common
template <typename T, std::int64_t dim, std::enable_if_t<dim == 1>* = nullptr>
inline auto make_device_ndarray(sycl::queue& q, const std::vector<T>& data) -> pr::ndarray<T, 1> {
    ONEDAL_ASSERT(data.size() > 0);
    const auto shape = pr::ndshape<1>{ std::int64_t(data.size()) };
    return pr::ndarray<T, 1>::wrap(data.data(), shape).to_device(q);
}

// TODO: Move to common
template <typename T, std::int64_t dim, std::enable_if_t<dim == 2>* = nullptr>
inline auto make_device_ndarray(sycl::queue& q, const std::vector<std::vector<T>>& data)
    -> pr::ndarray<T, 2> {
    ONEDAL_ASSERT(data.size() > 0);

    const std::size_t row_size = data.front().size();
    const auto shape = pr::ndshape<2>{ std::int64_t(data.size()), std::int64_t(row_size) };

    auto ary = pr::ndarray<T, 2>::empty(shape);
    T* ary_ptr = ary.get_mutable_data();

    for (const auto& row : data) {
        ONEDAL_ASSERT(row.size() == row_size);
        dal::backend::copy(ary_ptr, row.data(), std::int64_t(row_size));
        ary_ptr += row_size;
    }

    return ary.to_device(q);
}

template <typename TestType>
class empty_cluster_handling_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    pr::ndarray<float_t, 1> generate_uniform_host(std::int64_t count,
                                                  float_t a,
                                                  float_t b,
                                                  int seed = 7777) {
        ONEDAL_ASSERT(count > 0);
        ONEDAL_ASSERT(b > a);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float_t> distr(a, b);

        auto result = pr::ndarray<float_t, 1>::empty(count);
        float_t* result_ptr = result.get_mutable_data();

        for (std::int64_t i = 0; i < count; i++) {
            result_ptr[i] = distr(rng);
        }

        return result;
    }

    template <typename Integer>
    pr::ndarray<Integer, 1> generate_uniform_int_host(std::int64_t count,
                                                      Integer a,
                                                      Integer b,
                                                      int seed = 7777) {
        ONEDAL_ASSERT(count > 0);
        ONEDAL_ASSERT(b > a);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::int32_t> distr(a, b);

        auto result = pr::ndarray<Integer, 1>::empty(count);
        Integer* result_ptr = result.get_mutable_data();

        for (std::int64_t i = 0; i < count; i++) {
            result_ptr[i] = Integer(distr(rng));
        }

        return result;
    }

    pr::ndarray<std::int32_t, 1> generate_counters_host(std::int64_t cluster_count,
                                                        std::int64_t empty_cluster_count,
                                                        std::int64_t a,
                                                        std::int64_t b,
                                                        int seed = 7777) {
        ONEDAL_ASSERT(cluster_count > 0);
        ONEDAL_ASSERT(empty_cluster_count <= cluster_count);
        ONEDAL_ASSERT(a > 0);
        ONEDAL_ASSERT(b > a);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::int64_t> distr(a, b);

        auto counters = pr::ndarray<std::int32_t, 1>::empty(cluster_count);
        std::int32_t* counters_ptr = counters.get_mutable_data();

        for (std::int64_t el = 0; el < cluster_count; el++) {
            counters_ptr[el] = distr(rng);
        }

        auto indices_host = bk::make_unique_host<std::int32_t>(cluster_count);
        std::int32_t* indices_ptr = indices_host.get();
        for (std::int64_t i = 0; i < cluster_count; i++) {
            indices_ptr[i] = std::int32_t(i);
        }

        std::shuffle(indices_ptr, indices_ptr + cluster_count, rng);

        for (std::int64_t i = 0; i < empty_cluster_count; i++) {
            counters_ptr[indices_ptr[i]] = 0;
        }

        return counters;
    }

    pr::ndarray<float_t, 2> generate_data(std::int64_t row_count,
                                          std::int64_t column_count,
                                          int seed = 7777) {
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(column_count > 0);

        return generate_uniform_host(row_count * column_count, -5.0, 5.0, seed)
            .reshape(pr::ndshape<2>{ row_count, column_count })
            .to_device(this->get_queue());
    }

    pr::ndarray<float_t, 2> generate_closests_distances(std::int64_t cluster_count,
                                                        int seed = 7777) {
        return generate_uniform_host(cluster_count, 0.1, 10.0, seed)
            .reshape(pr::ndshape<2>{ cluster_count, 1 })
            .to_device(this->get_queue());
    }

    pr::ndarray<std::int32_t, 1> generate_counters(std::int64_t cluster_count,
                                                   std::int64_t empty_cluster_count,
                                                   std::int64_t a,
                                                   std::int64_t b,
                                                   int seed = 7777) {
        return generate_counters_host(cluster_count, empty_cluster_count, a, b, seed)
            .to_device(this->get_queue());
    }

    centroid_candidates<float_t> generate_candidates(std::int64_t sample_count,
                                                     std::int64_t cluster_count,
                                                     std::int64_t candidate_count,
                                                     int seed = 7777) {
        ONEDAL_ASSERT(sample_count > 0);
        ONEDAL_ASSERT(cluster_count > 0);
        ONEDAL_ASSERT(candidate_count > 0);
        ONEDAL_ASSERT(sample_count >= cluster_count);
        ONEDAL_ASSERT(cluster_count >= candidate_count);

        auto host_counters = generate_counters_host(cluster_count, candidate_count, 10, 100, seed);
        auto host_distances = generate_uniform_host(candidate_count, 0.1, 10.0, seed);
        auto host_indices = this->template generate_uniform_int_host<std::int32_t>( //
            candidate_count,
            0,
            sample_count - 1,
            seed);

        auto host_empty_cluster_indices = pr::ndarray<std::int32_t, 1>::empty(candidate_count);

        {
            const std::int32_t* host_counters_ptr = host_counters.get_data();
            std::int32_t* host_empty_cluster_indices_ptr =
                host_empty_cluster_indices.get_mutable_data();
            std::int64_t index = 0;

            for (std::int64_t i = 0; i < cluster_count; i++) {
                if (host_counters_ptr[i] > 0) {
                    continue;
                }

                host_empty_cluster_indices_ptr[index] = i;
                index++;
            }

            ONEDAL_ASSERT(index == candidate_count);
        }

        {
            float_t* host_distances_ptr = host_distances.get_mutable_data();
            std::sort(host_distances_ptr,
                      host_distances_ptr + candidate_count,
                      std::greater<float_t>{});
        }

        // Source cluster of each candidate row. `fill_empty_clusters` needs this populated in the
        // distributed path, where `reduce_candidates` shuffles it alongside the winning rows.
        auto host_source_clusters = this->template generate_uniform_int_host<std::int32_t>( //
            candidate_count,
            0,
            std::int32_t(cluster_count - 1),
            seed);

        return centroid_candidates<float_t>{ host_indices.to_device(this->get_queue()),
                                             host_distances.to_device(this->get_queue()),
                                             host_empty_cluster_indices.to_device(
                                                 this->get_queue()),
                                             host_source_clusters.to_device(this->get_queue()) };
    }

    pr::ndarray<std::int32_t, 2> generate_responses(std::int64_t row_count,
                                                    std::int64_t cluster_count,
                                                    int seed = 7777) {
        return this
            ->template generate_uniform_int_host<std::int32_t>(row_count,
                                                               0,
                                                               std::int32_t(cluster_count - 1),
                                                               seed)
            .reshape(pr::ndshape<2>{ row_count, 1 })
            .to_device(this->get_queue());
    }

    template <typename T, std::int64_t dim>
    auto make_partitions(const std::vector<std::vector<T>>& data,
                         const pr::ndshape<dim>& shape,
                         sycl::usm::alloc alloc_kind = sycl::usm::alloc::device)
        -> std::vector<pr::ndarray<T, dim>> {
        std::vector<pr::ndarray<T, dim>> partitions;
        partitions.reserve(data.size());

        for (const auto& entry : data) {
            ONEDAL_ASSERT(std::int64_t(entry.size()) == shape.get_count());
            auto partition = pr::ndarray<T, dim>::empty(this->get_queue(), shape, alloc_kind);
            partition.assign(this->get_queue(), entry.data(), shape.get_count()).wait_and_throw();
            partitions.push_back(std::move(partition));
        }

        return partitions;
    }

    std::vector<centroid_candidates<float_t>> make_candidate_partitions(
        const std::vector<pr::ndarray<std::int32_t, 1>>& indices,
        const std::vector<pr::ndarray<float_t, 1>>& distances,
        const std::vector<pr::ndarray<std::int32_t, 1>>& empty_centroid_indices,
        const std::vector<pr::ndarray<std::int32_t, 1>>& source_clusters) {
        ONEDAL_ASSERT(indices.size() == distances.size());
        ONEDAL_ASSERT(indices.size() == empty_centroid_indices.size());
        ONEDAL_ASSERT(indices.size() == source_clusters.size());

        std::vector<centroid_candidates<float_t>> candidate_partitions;
        candidate_partitions.reserve(indices.size());

        for (std::size_t i = 0; i < indices.size(); i++) {
            candidate_partitions.push_back(centroid_candidates<float_t>{ indices[i],
                                                                         distances[i],
                                                                         empty_centroid_indices[i],
                                                                         source_clusters[i] });
        }

        return candidate_partitions;
    }

    void run_find_candidates(const pr::ndarray<float_t, 2>& closest_distances,
                             const pr::ndarray<std::int32_t, 1>& counters,
                             const pr::ndarray<std::int32_t, 2>& responses,
                             std::int64_t candidate_count) {
        auto [candidates, find_candidates_event] = find_candidates( //
            this->get_queue(),
            candidate_count,
            closest_distances,
            counters,
            responses);
        find_candidates_event.wait_and_throw();

        check_candidates(closest_distances, candidates);
        check_source_clusters(responses, candidates);
    }

    void run_fill_empty_clusters(std::int64_t cluster_count,
                                 const pr::ndarray<float_t, 2>& data,
                                 centroid_candidates<float_t>& candidates) {
        ONEDAL_ASSERT(cluster_count > 0);
        const std::int64_t column_count = data.get_dimension(1);

        auto centroids = pr::ndarray<float_t, 2>::empty( //
            this->get_queue(),
            { cluster_count, column_count },
            sycl::usm::alloc::device);

        bk::communicator<spmd::device_memory_access::usm> fake_comm;
        fill_empty_clusters(this->get_queue(), fake_comm, data, candidates, centroids)
            .wait_and_throw();

        check_filled_centroids(data, candidates, centroids);
    }

    void run_fill_empty_clusters_distr(
        std::int64_t thread_count,
        const std::vector<pr::ndarray<float_t, 2>>& data_per_rank,
        std::vector<centroid_candidates<float_t>>& candidates_per_rank,
        const pr::ndarray<float_t, 2>& expected_centroids) {
        te::thread_communicator<spmd::device_memory_access::usm> thread_comm{ this->get_queue(),
                                                                              thread_count };
        bk::communicator<spmd::device_memory_access::usm> backend_comm{ thread_comm };

        const auto centroids_per_rank = thread_comm.map([&](std::int64_t rank) {
            auto [centroids, centroids_event] = pr::ndarray<float_t, 2>::zeros( //
                this->get_queue(),
                expected_centroids.get_shape(),
                sycl::usm::alloc::device);

            fill_empty_clusters(this->get_queue(),
                                backend_comm,
                                data_per_rank[rank],
                                candidates_per_rank[rank],
                                centroids,
                                { centroids_event })
                .wait_and_throw();

            return centroids;
        });

        for (std::int64_t rank = 0; rank < thread_count; rank++) {
            CAPTURE(rank);
            check_if_centroids_expected(expected_centroids, centroids_per_rank[rank]);
        }
    }

    void check_source_clusters(const pr::ndarray<std::int32_t, 2>& responses,
                               const centroid_candidates<float_t>& candidates) {
        const std::int64_t candidate_count = candidates.get_candidate_count();

        const auto host_responses = responses.to_host(this->get_queue());
        const auto host_indices = candidates.get_indices().to_host(this->get_queue());
        const auto host_sources = candidates.get_source_clusters().to_host(this->get_queue());

        const std::int32_t* host_responses_ptr = host_responses.get_data();
        const std::int32_t* host_indices_ptr = host_indices.get_data();
        const std::int32_t* host_sources_ptr = host_sources.get_data();

        for (std::int64_t i = 0; i < candidate_count; i++) {
            CAPTURE(i, host_indices_ptr[i], host_sources_ptr[i]);
            REQUIRE(host_sources_ptr[i] == host_responses_ptr[host_indices_ptr[i]]);
        }
    }

    void check_candidates(const pr::ndarray<float_t, 2>& closest_distances,
                          const centroid_candidates<float_t>& candidates) {
        const std::int64_t candidate_count = candidates.get_candidate_count();
        const std::int64_t elem_count = closest_distances.get_dimension(0);

        const auto host_closest_distances = closest_distances.to_host(this->get_queue());
        const auto host_candidate_indices = candidates.get_indices().to_host(this->get_queue());
        const auto host_candidate_distances = candidates.get_distances().to_host(this->get_queue());

        auto closest_distances_ptr = host_closest_distances.get_data();
        auto candidate_indices_ptr = host_candidate_indices.get_data();
        auto candidate_distances_ptr = host_candidate_distances.get_data();

        for (std::int64_t i = 0; i < candidate_count; i++) {
            auto distance = candidate_distances_ptr[i];
            auto index = candidate_indices_ptr[i];
            std::int64_t count = 0;
            for (std::int64_t j = 0; j < elem_count; j++) {
                auto cur_val = -1.0 * closest_distances_ptr[j];
                if (cur_val >= distance)
                    count++;
            }
            CAPTURE(i, count);
            REQUIRE(count <= candidate_count);
            REQUIRE(index >= 0);
            REQUIRE(index < elem_count);
        }
    }

    void check_filled_centroids(const pr::ndarray<float_t, 2>& data,
                                const centroid_candidates<float_t>& candidates,
                                const pr::ndarray<float_t, 2>& centroids) {
        const std::int64_t column_count = data.get_dimension(1);
        const std::int64_t candidate_count = candidates.get_candidate_count();

        const auto host_data = data.to_host(this->get_queue());
        const auto host_centroids = centroids.to_host(this->get_queue());
        const auto host_candidate_indices = candidates.get_indices().to_host(this->get_queue());
        const auto host_empty_cluster_indices =
            candidates.get_empty_cluster_indices().to_host(this->get_queue());

        const float_t* host_data_ptr = host_data.get_data();
        const float_t* host_centroids_ptr = host_centroids.get_data();
        const std::int32_t* host_candidate_indices_ptr = host_candidate_indices.get_data();
        const std::int32_t* host_empty_cluster_indices_ptr = host_empty_cluster_indices.get_data();

        for (std::int64_t i = 0; i < candidate_count; i++) {
            const float_t* data_vector =
                host_data_ptr + host_candidate_indices_ptr[i] * column_count;
            const float_t* centroid =
                host_centroids_ptr + host_empty_cluster_indices_ptr[i] * column_count;

            CAPTURE(i, host_candidate_indices_ptr[i], host_empty_cluster_indices_ptr[i]);
            check_if_feature_vectors_equal(data_vector, centroid, column_count);
        }
    }

    void check_if_centroids_expected(const pr::ndarray<float_t, 2>& expected_centroids,
                                     const pr::ndarray<float_t, 2>& actual_centroids) {
        REQUIRE(actual_centroids.get_shape() == expected_centroids.get_shape());

        const std::int64_t column_count = expected_centroids.get_dimension(1);
        const auto host_actual_centroids = actual_centroids.to_host(this->get_queue());
        const auto host_expected_centroids = expected_centroids.to_host(this->get_queue());

        const float_t* host_actual_centroids_ptr = host_actual_centroids.get_data();
        const float_t* host_expected_centroids_ptr = host_expected_centroids.get_data();

        for (std::int64_t i = 0; i < expected_centroids.get_dimension(0); i++) {
            CAPTURE(i);
            check_if_feature_vectors_equal(host_expected_centroids_ptr + i * column_count,
                                           host_actual_centroids_ptr + i * column_count,
                                           column_count);
        }
    }

    void check_if_feature_vectors_equal(const float_t* expected,
                                        const float_t* actual,
                                        std::int64_t count,
                                        float_t tol = 1.0e-5) {
        for (std::int64_t j = 0; j < count; j++) {
            auto maxv = std::max(std::fabs(expected[j]), std::fabs(actual[j]));
            if (maxv == 0.0) {
                continue;
            }
            CAPTURE(j, expected[j], actual[j]);
            REQUIRE(std::fabs(expected[j] - actual[j]) / maxv < tol);
        }
    }
};

using kmeans_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(empty_cluster_handling_test, "find candidates", "[candidates]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    using config_t = std::tuple<std::int64_t, std::int64_t>;

    const auto [cluster_count, candidate_count] = GENERATE( //
        config_t{ 11, 2 },
        config_t{ 1001, 17 });

    const auto closest_distances = this->generate_closests_distances(cluster_count);
    const auto counters = this->generate_counters(cluster_count, candidate_count, 10, 100);
    // `find_candidates` reads the source cluster of every candidate row out of the responses,
    // so the responses must be as long as `closest_distances`.
    const auto responses = this->generate_responses(cluster_count, cluster_count);

    this->run_find_candidates(closest_distances, counters, responses, candidate_count);
}

TEMPLATE_LIST_TEST_M(empty_cluster_handling_test,
                     "fill empty clusters local",
                     "[fill]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    const std::int64_t row_count = 56;
    const std::int64_t column_count = 5;
    const std::int64_t cluster_count = 11;
    const std::int64_t candidate_count = 5;

    const auto data = this->generate_data(row_count, column_count);
    auto candidates = this->generate_candidates(row_count, cluster_count, candidate_count);

    this->run_fill_empty_clusters(cluster_count, data, candidates);
}

TEMPLATE_LIST_TEST_M(empty_cluster_handling_test,
                     "fill empty clusters distributed",
                     "[fill][distr]",
                     kmeans_types) {
    using float_t = TestType;
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    const std::int64_t thread_count = 2;

    const auto data = //
        make_device_ndarray<float_t, 2>( //
            this->get_queue(),
            { { 11.1, 11.2, 11.3 },
              { 12.1, 12.2, 12.3 },
              { 13.1, 13.2, 13.3 },
              { 21.1, 21.2, 21.3 },
              { 22.1, 22.2, 22.3 },
              { 23.1, 23.2, 23.3 } })
            .split(thread_count);

    const auto candidate_indices = //
        make_device_ndarray<std::int32_t, 1>( //
            this->get_queue(), //
            { 0, 2, 1, 2 })
            .split(thread_count);

    const auto candidate_distances = //
        make_device_ndarray<float_t, 1>( //
            this->get_queue(), //
            { 10.3, 5.7, 7.3, 5.9 })
            .split(thread_count);

    // Empty cluster indices must be the same for all ranks
    const auto candidate_empty_cluster_indices = //
        make_device_ndarray<std::int32_t, 1>( //
            this->get_queue(), //
            { 0, 2, 0, 2 })
            .split(thread_count);

    // Source cluster of every candidate row on the emitting rank. `reduce_candidates` shuffles it
    // alongside the winning rows, so it has to be provided per rank as well.
    const auto candidate_source_clusters = //
        make_device_ndarray<std::int32_t, 1>( //
            this->get_queue(), //
            { 1, 1, 1, 1 })
            .split(thread_count);

    auto candidates = this->make_candidate_partitions(candidate_indices,
                                                      candidate_distances,
                                                      candidate_empty_cluster_indices,
                                                      candidate_source_clusters);

    const auto expected_centroids = //
        make_device_ndarray<float_t, 2>( //
            this->get_queue(),
            { { 11.1, 11.2, 11.3 }, //
              { 0.00, 0.00, 0.00 }, //
              { 22.1, 22.2, 22.3 } });

    this->run_fill_empty_clusters_distr(thread_count, data, candidates, expected_centroids);
}

} // namespace oneapi::dal::kmeans::backend::test
