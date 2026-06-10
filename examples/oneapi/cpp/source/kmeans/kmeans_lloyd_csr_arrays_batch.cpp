/*******************************************************************************
* Copyright contributors to the oneDAL project
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

// Reproducer for sparse-KMeans uninitialized-memory issue (DAALL-9413).
//
// Unlike kmeans_lloyd_csr_batch.cpp, this example does NOT read the CSR data
// through dal::csv::data_source. Instead it constructs the dal::csr_table
// from dal::array<float>/dal::array<int64_t> objects that were filled by user
// code. This is exactly the path that scikit-learn-intelex / daal4py exercise
// when handing a scipy.sparse.csr_matrix to oneDAL, and the path that triggers
// MSan errors at host_csr_table_adapter.cpp:161 / setArrays<float>.

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace dal = oneapi::dal;

namespace {

// Build a synthetic CSR matrix with `row_count` rows, `col_count` columns and
// `nnz_per_row` non-zeros per row, all in 1-based indexing (which is what
// daal4py forwards from scipy when the matrix has been base-1 converted).
template <typename Float>
struct csr_blob {
    dal::array<Float> values;
    dal::array<std::int64_t> column_indices;
    dal::array<std::int64_t> row_offsets;
    std::int64_t row_count;
    std::int64_t column_count;
};

template <typename Float>
csr_blob<Float> make_csr(std::int64_t row_count,
                         std::int64_t column_count,
                         std::int64_t nnz_per_row,
                         std::uint64_t seed,
                         bool one_based) {
    const std::int64_t nnz = row_count * nnz_per_row;
    auto values = dal::array<Float>::empty(nnz);
    auto column_indices = dal::array<std::int64_t>::empty(nnz);
    auto row_offsets = dal::array<std::int64_t>::empty(row_count + 1);

    std::mt19937_64 gen{ seed };
    std::uniform_real_distribution<Float> val_dist{ Float(0.1), Float(1.0) };

    Float* values_ptr = values.get_mutable_data();
    std::int64_t* column_indices_ptr = column_indices.get_mutable_data();
    std::int64_t* row_offsets_ptr = row_offsets.get_mutable_data();

    const std::int64_t base_index = one_based ? 1 : 0;
    row_offsets_ptr[0] = base_index;
    for (std::int64_t r = 0; r < row_count; r++) {
        const std::int64_t base = r * nnz_per_row;
        const std::int64_t step = column_count / nnz_per_row;
        for (std::int64_t k = 0; k < nnz_per_row; k++) {
            column_indices_ptr[base + k] = base_index + k * step + (r % step);
            values_ptr[base + k] = val_dist(gen);
        }
        row_offsets_ptr[r + 1] = row_offsets_ptr[r] + nnz_per_row;
    }

    return { std::move(values),
             std::move(column_indices),
             std::move(row_offsets),
             row_count,
             column_count };
}

template <typename Float>
dal::csr_table make_csr_table(const csr_blob<Float>& b, bool one_based) {
    return dal::csr_table::wrap(
        b.values,
        b.column_indices,
        b.row_offsets,
        b.column_count,
        one_based ? dal::sparse_indexing::one_based : dal::sparse_indexing::zero_based);
}

template <typename Float>
dal::table make_initial_centroids(std::int64_t cluster_count, std::int64_t column_count) {
    auto centroids = dal::array<Float>::empty(cluster_count * column_count);
    Float* p = centroids.get_mutable_data();
    std::mt19937_64 gen{ 42 };
    std::uniform_real_distribution<Float> dist{ Float(0), Float(1) };
    for (std::int64_t i = 0; i < cluster_count * column_count; i++) {
        p[i] = dist(gen);
    }
    return dal::homogen_table::wrap(centroids, cluster_count, column_count);
}

} // namespace

int main(int argc, char const* argv[]) {
    using Float = float;

    constexpr std::int64_t row_count = 1024;
    constexpr std::int64_t column_count = 32;
    constexpr std::int64_t nnz_per_row = 4;
    constexpr std::int64_t cluster_count = 8;

    // Toggle one-based vs zero-based via argv[1]; defaults to one-based since
    // that is what daal4py / scipy.sparse forwards.
    const bool one_based = !(argc >= 2 && std::string(argv[1]) == "zero");
    std::cout << "indexing: " << (one_based ? "one_based" : "zero_based") << std::endl;

    // Build the CSR table from user arrays. This is the path that triggers
    // host_csr_table_adapter::host_csr_table_adapter -> setArrays<float>.
    const auto blob = make_csr<Float>(row_count, column_count, nnz_per_row, 17, one_based);
    const auto x_train = make_csr_table(blob, one_based);
    const auto initial_centroids = make_initial_centroids<Float>(cluster_count, column_count);

    const auto kmeans_desc = dal::kmeans::descriptor<Float, dal::kmeans::method::lloyd_csr>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(Float(0.001));

    // Run train + infer in a loop so MSan / ASan get many chances to flag the
    // issue, and so we can see if the result drifts across runs of an
    // identical fit. With a deterministic kernel + fixed inputs the obj fn
    // value should be bit-exact across all trials. Drift => uninitialized
    // memory feeding the kernel.
    double first_obj = 0.0;
    bool drift = false;
    for (int trial = 0; trial < 32; trial++) {
        const auto train_result = dal::train(kmeans_desc, x_train, initial_centroids);
        const auto infer_result = dal::infer(kmeans_desc, train_result.get_model(), x_train);
        const double obj = train_result.get_objective_function_value();
        if (trial == 0) {
            first_obj = obj;
            std::cout << "Trial 0: iter=" << train_result.get_iteration_count() << " obj=" << obj
                      << std::endl;
        }
        else if (obj != first_obj) {
            std::cout << "Trial " << trial << " DRIFT: obj=" << obj
                      << " (delta=" << (obj - first_obj) << ")" << std::endl;
            drift = true;
        }
        if (train_result.get_iteration_count() < 0) {
            std::cerr << "negative iteration count!" << std::endl;
            return 1;
        }
        (void)infer_result;
    }

    std::cout << (drift ? "DRIFT detected" : "Stable") << std::endl;
    return drift ? 2 : 0;
}
