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

#include <limits>

#include "oneapi/dal/test/engine/fixtures.hpp"

#include "oneapi/dal/algo/hdbscan/backend/gpu/kernel_impl.hpp"

namespace oneapi::dal::hdbscan::backend::test {

// The sizing helper guards row counts whose n x n matrix is several gigabytes, so it is
// driven directly rather than through `dal::compute`.
TEST("hdbscan gpu: brute_force rejects a matrix larger than device memory", "[hdbscan][gpu]") {
    constexpr std::int64_t gib = 1024 * 1024 * 1024;
    constexpr std::int64_t f64 = 8;
    // The simd k-selection path, which keeps no device scratch of its own.
    constexpr std::int64_t k = 5;
    constexpr std::int64_t no_scratch = 0;

    // 16 GiB global, 4 GiB per allocation: 10k rows need 800 MB and fit.
    REQUIRE(backend::mrd_matrix_fits_on_device(10000, 8, f64, 4 * gib, 16 * gib, k, no_scratch));

    // 30k rows need 7.2 GB, past the single-allocation limit.
    REQUIRE(!backend::mrd_matrix_fits_on_device(30000, 8, f64, 4 * gib, 16 * gib, k, no_scratch));

    // A 32k x 32k matrix is 7.63 GiB and fits both one allocation and the 9 GiB
    // budget, but the input table pushes the total over once it gets wide.
    REQUIRE(backend::mrd_matrix_fits_on_device(32000, 8, f64, 16 * gib, 18 * gib, k, no_scratch));
    REQUIRE(
        !backend::mrd_matrix_fits_on_device(32000, 8000, f64, 16 * gib, 18 * gib, k, no_scratch));

    // 100k float64 rows need 80 GB, more than a 48 GB card.
    REQUIRE(!backend::mrd_matrix_fits_on_device(100000, 8, f64, 48 * gib, 48 * gib, k, no_scratch));

    // Only part of global memory is usable: a 32.5 GiB matrix allocates on a
    // 48 GiB card and then faults on first access, so it has to be rejected
    // even though it is under the 45 GiB single-allocation limit.
    REQUIRE(!backend::mrd_matrix_fits_on_device(66000, 8, f64, 45 * gib, 48 * gib, k, no_scratch));
    REQUIRE(backend::mrd_matrix_fits_on_device(50000, 8, f64, 45 * gib, 48 * gib, k, no_scratch));

    // Halving the element size halves the footprint, so float32 is checked
    // against its own size rather than the widest one.
    REQUIRE(backend::mrd_matrix_fits_on_device(20000, 8, 4, 4 * gib, 16 * gib, k, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(20000, 8, f64, 2 * gib, 16 * gib, k, no_scratch));

    // A footprint that leaves int64 saturates instead of wrapping negative:
    // `8 * 2^30 * 2^30` is exactly 2^63.
    constexpr std::int64_t pow30 = std::int64_t(1) << 30;
    REQUIRE(!backend::mrd_matrix_fits_on_device(pow30, 8, f64, 45 * gib, 48 * gib, k, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(std::int64_t(1) << 40,
                                                8,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                k,
                                                no_scratch));
    // The column count, `min_samples` and the reported scratch each overflow on
    // their own.
    REQUIRE(!backend::mrd_matrix_fits_on_device(1000,
                                                std::int64_t(1) << 50,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                k,
                                                no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(1000,
                                                std::numeric_limits<std::int64_t>::max(),
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                k,
                                                no_scratch));
    REQUIRE(
        !backend::mrd_matrix_fits_on_device(50000, 8, f64, 45 * gib, 48 * gib, pow30, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(50000,
                                                8,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                k,
                                                backend::mrd_bytes_saturated));

    // 55k float64 rows fit the matrix and the budget on a 48 GiB card, but not
    // once quick select's two n x n scratch matrices are counted.
    constexpr std::int64_t quick_rows = 55000;
    constexpr std::int64_t quick_scratch = quick_rows * quick_rows * (f64 + 4) + 1024 * f64;
    REQUIRE(
        backend::mrd_matrix_fits_on_device(quick_rows, 8, f64, 45 * gib, 48 * gib, k, no_scratch));
    REQUIRE(!backend::mrd_matrix_fits_on_device(quick_rows,
                                                8,
                                                f64,
                                                45 * gib,
                                                48 * gib,
                                                1023,
                                                quick_scratch));

    // The `n x min_samples` selection output counts on every path, quick or not.
    REQUIRE(
        !backend::mrd_matrix_fits_on_device(50000, 8, f64, 45 * gib, 48 * gib, 20000, no_scratch));
}

TEST("hdbscan gpu: compute_core_distances returns a completed event", "[hdbscan][gpu]") {
    namespace pr = dal::backend::primitives;
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    constexpr std::int64_t n = 64;
    constexpr std::int64_t min_samples = 5;

    auto [dist, dist_event] =
        pr::ndarray<float, 2>::full(q, { n, n }, 1.0f, sycl::usm::alloc::device);
    auto [core, core_event] = pr::ndarray<float, 1>::zeros(q, n, sycl::usm::alloc::device);

    pr::ndview<float, 1> core_view = core;
    const auto event = backend::compute_core_distances<float>(q,
                                                              dist,
                                                              core_view,
                                                              min_samples,
                                                              n,
                                                              distance_metric::manhattan,
                                                              { dist_event, core_event });
    // The k-selection scratch the kernel reads is freed on return, so the event must be done.
    const auto status = event.get_info<sycl::info::event::command_execution_status>();
    REQUIRE(status == sycl::info::event_command_status::complete);

    const auto core_host = core.to_host(q);
    for (std::int64_t i = 0; i < n; ++i) {
        REQUIRE(core_host.get_data()[i] == 1.0f);
    }
}

} // namespace oneapi::dal::hdbscan::backend::test
