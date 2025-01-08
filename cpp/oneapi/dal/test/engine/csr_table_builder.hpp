/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::test::engine {

template <typename Float = float>
csr_table copy_data_to_csr(const dal::array<Float>& data,
                           const dal::array<std::int64_t>& column_indices,
                           const dal::array<std::int64_t>& row_offsets,
                           const sparse_indexing indexing,
                           const std::int64_t column_count,
                           const std::int64_t row_count) {
    auto row_offs_ptr = row_offsets.get_data();
    auto data_ptr = data.get_data();
    auto col_indices_ptr = column_indices.get_data();
    auto nnz_count = row_offs_ptr[row_count] - row_offs_ptr[0];
    const auto copied_data = dal::array<Float>::empty(nnz_count);
    const auto copied_col_indices = dal::array<std::int64_t>::empty(nnz_count);
    const auto copied_row_offsets = dal::array<std::int64_t>::empty(row_count + 1);

    auto copied_data_ptr = copied_data.get_mutable_data();
    auto copied_col_indices_ptr = copied_col_indices.get_mutable_data();
    auto copied_row_offsets_ptr = copied_row_offsets.get_mutable_data();
    for (std::int32_t i = 0; i < nnz_count; ++i) {
        copied_data_ptr[i] = data_ptr[i];
        copied_col_indices_ptr[i] = col_indices_ptr[i];
    }
    for (std::int32_t i = 0; i <= row_count; ++i) {
        copied_row_offsets_ptr[i] = row_offs_ptr[i];
    }
    return csr_table::wrap(copied_data,
                           copied_col_indices,
                           copied_row_offsets,
                           column_count,
                           indexing);
}

template <typename Float = float>
const dal::array<Float> copy_data_to_dense_array(const dal::array<Float>& data,
                                                 const dal::array<std::int64_t>& column_indices,
                                                 const dal::array<std::int64_t>& row_offsets,
                                                 const sparse_indexing indexing,
                                                 const std::int64_t column_count,
                                                 const std::int64_t row_count) {
    using Index = std::int64_t;
    auto dense_data_host = dal::array<Float>::zeros(row_count * column_count);
    auto dense_data_ptr = dense_data_host.get_mutable_data();

    const auto data_ptr = data.get_data();
    const auto col_ind_ptr = column_indices.get_data();
    const auto row_offs_ptr = row_offsets.get_data();
    const Index shift = bool(indexing == sparse_indexing::one_based);
    for (Index row_idx = 0; row_idx < row_count; ++row_idx) {
        const Index start = row_offs_ptr[row_idx] - shift;
        const Index end = row_offs_ptr[row_idx + 1] - shift;
        for (Index data_idx = start; data_idx < end; ++data_idx) {
            auto col_idx = col_ind_ptr[data_idx] - shift;
            dense_data_ptr[row_idx * column_count + col_idx] = data_ptr[data_idx];
        }
    }
    return dense_data_host;
}

template <typename Float = float>
homogen_table copy_data_to_dense(const dal::array<Float>& data,
                                 const dal::array<std::int64_t>& column_indices,
                                 const dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing indexing,
                                 const std::int64_t column_count,
                                 const std::int64_t row_count) {
    auto dense_data_host = copy_data_to_dense_array(data,
                                                    column_indices,
                                                    row_offsets,
                                                    indexing,
                                                    column_count,
                                                    row_count);
    return homogen_table::wrap(dense_data_host, row_count, column_count);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Float = float>
csr_table copy_data_to_csr(sycl::queue& queue,
                           const dal::array<Float>& data,
                           const dal::array<std::int64_t>& column_indices,
                           const dal::array<std::int64_t>& row_offsets,
                           const sparse_indexing indexing,
                           const std::int64_t column_count,
                           const std::int64_t row_count) {
    auto row_offs_ptr = row_offsets.get_data();
    auto nnz_count = row_offs_ptr[row_count] - row_offs_ptr[0];
    const auto copied_data = dal::array<Float>::empty(queue, nnz_count, sycl::usm::alloc::device);
    const auto copied_col_indices =
        dal::array<std::int64_t>::empty(queue, nnz_count, sycl::usm::alloc::device);
    const auto copied_row_offsets =
        dal::array<std::int64_t>::empty(queue, row_count + 1, sycl::usm::alloc::device);
    auto data_event = queue.copy<Float>(data.get_data(), copied_data.get_mutable_data(), nnz_count);
    auto col_indices_event = queue.copy<std::int64_t>(column_indices.get_data(),
                                                      copied_col_indices.get_mutable_data(),
                                                      nnz_count);
    auto row_offsets_event = queue.copy<std::int64_t>(row_offsets.get_data(),
                                                      copied_row_offsets.get_mutable_data(),
                                                      row_count + 1);
    sycl::event::wait_and_throw({ data_event, col_indices_event, row_offsets_event });
    return csr_table::wrap(copied_data,
                           copied_col_indices,
                           copied_row_offsets,
                           column_count,
                           indexing);
}

template <typename Float = float>
homogen_table copy_data_to_dense(sycl::queue& queue,
                                 const dal::array<Float>& data,
                                 const dal::array<std::int64_t>& column_indices,
                                 const dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing indexing,
                                 const std::int64_t column_count,
                                 const std::int64_t row_count) {
    auto dense_data_host = copy_data_to_dense_array(data,
                                                    column_indices,
                                                    row_offsets,
                                                    indexing,
                                                    column_count,
                                                    row_count);
    const auto dense_data_device =
        dal::array<Float>::empty(queue, row_count * column_count, sycl::usm::alloc::device);
    queue
        .copy<Float>(dense_data_host.get_data(),
                     dense_data_device.get_mutable_data(),
                     row_count * column_count)
        .wait_and_throw();
    return homogen_table::wrap(dense_data_device, row_count, column_count);
}
#endif // ONEDAL_DATA_PARALLEL

/**
* Generates random CSR table based on inputs
*/
template <typename Float = float>
struct csr_table_builder {
    std::int64_t row_count_, column_count_;
    float nonzero_fraction_;
    sparse_indexing indexing_;
    const dal::array<Float> data_;
    const dal::array<std::int64_t> column_indices_;
    const dal::array<std::int64_t> row_offsets_;

    csr_table_builder(std::int64_t row_count,
                      std::int64_t column_count,
                      float nnz_fraction = 0.1,
                      sparse_indexing indexing = sparse_indexing::one_based,
                      float min_val = -10.0,
                      float max_val = 10.0,
                      int seed_in = 42)
            : row_count_(row_count),
              column_count_(column_count),
              nonzero_fraction_(nnz_fraction),
              indexing_(indexing),
              data_(dal::array<Float>::empty(nnz_fraction * row_count * column_count)),
              column_indices_(
                  dal::array<std::int64_t>::empty(nnz_fraction * row_count * column_count)),
              row_offsets_(dal::array<std::int64_t>::empty(row_count + 1)) {
        std::int64_t total_count = row_count_ * column_count_;
        std::int64_t nonzero_count = total_count * nnz_fraction;
        std::int64_t indexing_shift = bool(indexing == sparse_indexing::one_based);

        std::uint32_t seed = seed_in;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> uniform_data(min_val, max_val);
        std::uniform_int_distribution<std::int64_t> uniform_indices(
            0,
            column_count_ - 1 - indexing_shift);
        std::uniform_int_distribution<std::int64_t> uniform_ind_count(1, column_count_ - 2);

        auto data_ptr = data_.get_mutable_data();
        auto col_indices_ptr = column_indices_.get_mutable_data();
        auto row_offsets_ptr = row_offsets_.get_mutable_data();
        // Generate data
        for (std::int32_t i = 0; i < nonzero_count; ++i) {
            data_ptr[i] = uniform_data(rng);
        }
        // Generate column indices and fill row offsets
        std::int64_t row_idx = 0;
        std::int64_t fill_count = 0;
        row_offsets_ptr[0] = indexing_shift;
        while (fill_count < nonzero_count && row_idx < row_count_) {
            // Generate the number of non-zero columns for current row
            std::int64_t nnz_col_count = uniform_ind_count(rng);
            nnz_col_count = std::min(nnz_col_count, nonzero_count - fill_count);
            for (std::int32_t i = 0; i < nnz_col_count; ++i) {
                std::int64_t col_idx = uniform_indices(rng) + indexing_shift;
                col_indices_ptr[fill_count + i] = col_idx + indexing_shift;
            }
            std::sort(col_indices_ptr + fill_count, col_indices_ptr + fill_count + nnz_col_count);
            // Remove duplications
            std::int64_t dup_count = 0;
            for (std::int32_t i = 1; i < nnz_col_count; ++i) {
                auto cur_ptr = col_indices_ptr + (fill_count - dup_count);
                if (cur_ptr[i] == cur_ptr[i - 1]) {
                    ++dup_count;
                    // Shift the tail if there is duplication
                    for (std::int32_t j = i + 1; j < nnz_col_count; ++j) {
                        cur_ptr[j - 1] = cur_ptr[j];
                    }
                }
            }
            fill_count += (nnz_col_count - dup_count);
            // Update row offsets
            row_offsets_ptr[row_idx + 1] = fill_count + indexing_shift;
            row_idx++;
        }
        if (row_idx < row_count_) {
            for (std::int32_t i = row_idx; i <= row_count_; ++i) {
                row_offsets_ptr[i] = fill_count + indexing_shift;
            }
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    csr_table build_csr_table(device_test_policy& policy) const {
        auto queue = policy.get_queue();
        return copy_data_to_csr(queue,
                                data_,
                                column_indices_,
                                row_offsets_,
                                indexing_,
                                column_count_,
                                row_count_);
    }

    homogen_table build_dense_table(device_test_policy& policy) const {
        auto queue = policy.get_queue();
        return copy_data_to_dense(queue,
                                  data_,
                                  column_indices_,
                                  row_offsets_,
                                  indexing_,
                                  column_count_,
                                  row_count_);
    }
#endif // ONEDAL_DATA_PARALLEL

    csr_table build_csr_table(host_test_policy& policy) const {
        return copy_data_to_csr(data_,
                                column_indices_,
                                row_offsets_,
                                indexing_,
                                column_count_,
                                row_count_);
    }

    homogen_table build_dense_table(host_test_policy& policy) const {
        return copy_data_to_dense(data_,
                                  column_indices_,
                                  row_offsets_,
                                  indexing_,
                                  column_count_,
                                  row_count_);
    }
};

/// Generates CSR table with clustering dataset.
/// Dataset is looks like multidimensional blobs
/// with fixed centroid and randomized points around centroid
/// with radius :expr:`r=1.0`.
template <typename Float = float>
struct csr_make_blobs {
    /// Indexing type used for generation
    using Index = std::int64_t;
    /// Dataset paramters
    Index row_count_, column_count_, cluster_count_;
    float nonzero_fraction_;
    sparse_indexing indexing_;
    const dal::array<Float> data_;
    const dal::array<Index> column_indices_;
    const dal::array<Index> row_offsets_;
    /// Dataset generation parameters
    const Float centroid_fill_value = 10.0f;
    const Float min_val = -1.0f;
    const Float max_val = 1.0f;

    csr_make_blobs(Index cluster_count,
                   Index row_count,
                   Index column_count,
                   float nnz_fraction = 0.05,
                   sparse_indexing indexing = sparse_indexing::one_based,
                   Index seed = 42)
            : row_count_(row_count),
              column_count_(column_count),
              cluster_count_(cluster_count),
              nonzero_fraction_(nnz_fraction),
              indexing_(indexing),
              data_(dal::array<Float>::empty(nnz_fraction * row_count * column_count)),
              column_indices_(dal::array<Index>::empty(nnz_fraction * row_count * column_count)),
              row_offsets_(dal::array<Index>::empty(row_count + 1)) {
        // Get data arrays
        auto data_ptr = data_.get_mutable_data();
        auto col_indices_ptr = column_indices_.get_mutable_data();
        auto row_offs_ptr = row_offsets_.get_mutable_data();
        const Index indexing_shift = bool(indexing == sparse_indexing::one_based);
        // Estimate number of non-zero values in each row
        const Index row_nonzero_count = column_count * nnz_fraction;
        // Init random engines
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> uniform_data(min_val, max_val);
        std::uniform_int_distribution<Index> uniform_indices(indexing_shift,
                                                             column_count + indexing_shift - 1);
        // Check if it is possible to generate non-empty row
        if (row_nonzero_count < 1) {
            std::cout << "ERROR: Non-zero fraction is too small to generate rows" << std::endl;
            ONEDAL_ASSERT(row_nonzero_count >= 1);
            return;
        }
        Index fill_count = 0;
        row_offs_ptr[0] = indexing_shift;
        // Create centroids
        for (Index cent_idx = 0; cent_idx < cluster_count; ++cent_idx) {
            std::set<Index> columns;
            while (Index(columns.size()) < row_nonzero_count) {
                const Index col_idx = uniform_indices(rng);
                columns.insert(col_idx);
            }
            for (auto iter = columns.begin(); iter != columns.end(); iter++) {
                data_ptr[fill_count] = centroid_fill_value * (cent_idx + 1);
                col_indices_ptr[fill_count] = *iter;
                fill_count++;
            }
            row_offs_ptr[cent_idx + 1] = fill_count + indexing_shift;
        }

        // Generate remaining rows adding random noise to centroids
        for (Index row_idx = cluster_count; row_idx < row_count; ++row_idx) {
            const Index centroid_id = row_idx % cluster_count;
            for (Index data_idx = row_offs_ptr[centroid_id] - indexing_shift;
                 data_idx < row_offs_ptr[centroid_id + 1] - indexing_shift;
                 ++data_idx) {
                col_indices_ptr[fill_count] = col_indices_ptr[data_idx];
                data_ptr[fill_count] = data_ptr[data_idx] + uniform_data(rng);
                fill_count++;
            }
            row_offs_ptr[row_idx + 1] = fill_count + indexing_shift;
        }
    }

    table get_data(host_test_policy& policy) const {
        return copy_data_to_csr(data_,
                                column_indices_,
                                row_offsets_,
                                indexing_,
                                column_count_,
                                row_count_);
    }

    table get_dense_data(host_test_policy& policy) const {
        return copy_data_to_dense(data_,
                                  column_indices_,
                                  row_offsets_,
                                  indexing_,
                                  column_count_,
                                  row_count_);
    }

#ifdef ONEDAL_DATA_PARALLEL
    table get_data(device_test_policy& policy) const {
        return copy_data_to_csr(policy.get_queue(),
                                data_,
                                column_indices_,
                                row_offsets_,
                                indexing_,
                                column_count_,
                                row_count_);
    }

    table get_dense_data(device_test_policy& policy) const {
        return copy_data_to_dense(policy.get_queue(),
                                  data_,
                                  column_indices_,
                                  row_offsets_,
                                  indexing_,
                                  column_count_,
                                  row_count_);
    }
#endif

    table get_initial_centroids() const {
        const auto result = dal::array<Float>::zeros(cluster_count_ * column_count_);
        auto result_ptr = result.get_mutable_data();

        const Index shift = bool(indexing_ == sparse_indexing::one_based);
        const auto data_ptr = data_.get_data();
        const auto col_ind_ptr = column_indices_.get_data();
        const auto row_offs_ptr = row_offsets_.get_data();
        for (Index row_idx = 0; row_idx < cluster_count_; ++row_idx) {
            const auto start = row_offs_ptr[row_idx] - shift;
            const auto end = row_offs_ptr[row_idx + 1] - shift;
            for (Index data_idx = start; data_idx < end; ++data_idx) {
                auto col_idx = col_ind_ptr[data_idx] - shift;
                result_ptr[row_idx * column_count_ + col_idx] = data_ptr[data_idx];
            }
        }
        return homogen_table::wrap(result, cluster_count_, column_count_);
    }

    table get_result_centroids() const {
        const auto result = dal::array<Float>::empty(cluster_count_ * column_count_);
        auto result_ptr = result.get_mutable_data();
        const auto cluster_counts = dal::array<std::int32_t>::empty(cluster_count_);
        auto counts_ptr = cluster_counts.get_mutable_data();

        const Index shift = bool(indexing_ == sparse_indexing::one_based);
        const auto data_ptr = data_.get_data();
        const auto col_ind_ptr = column_indices_.get_data();
        const auto row_offs_ptr = row_offsets_.get_data();
        for (Index row_idx = 0; row_idx < cluster_count_; ++row_idx) {
            counts_ptr[row_idx] = 0;
            for (Index col_id = 0; col_id < column_count_; ++col_id) {
                result_ptr[row_idx * column_count_ + col_id] = 0;
            }
        }
        for (Index row_idx = 0; row_idx < row_count_; ++row_idx) {
            const auto start = row_offs_ptr[row_idx] - shift;
            const auto end = row_offs_ptr[row_idx + 1] - shift;
            for (Index data_idx = start; data_idx < end; ++data_idx) {
                auto col_idx = col_ind_ptr[data_idx] - shift;
                result_ptr[(row_idx % cluster_count_) * column_count_ + col_idx] +=
                    data_ptr[data_idx];
            }
            counts_ptr[row_idx % cluster_count_]++;
        }
        for (Index row_idx = 0; row_idx < cluster_count_; ++row_idx) {
            for (Index col_id = 0; col_id < column_count_; ++col_id) {
                result_ptr[row_idx * column_count_ + col_id] /= counts_ptr[row_idx];
            }
        }
        return homogen_table::wrap(result, cluster_count_, column_count_);
    }

    table get_responses() const {
        auto responses = dal::array<std::int32_t>::empty(row_count_);
        auto response_ptr = responses.get_mutable_data();
        for (std::int32_t i = 0; i < row_count_; ++i) {
            response_ptr[i] = i % cluster_count_;
        }
        return homogen_table::wrap(response_ptr, row_count_, 1);
    }
};

} //namespace oneapi::dal::test::engine
