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

// HDBSCAN tunable playground (DPC++ / GPU + CPU SYCL).
//
// Mirrors the host playground: synthetic 2-D Gaussian blobs + uniform noise
// fed through HDBSCAN parameter sweeps so changes in input or parameters
// translate directly into observable changes in output.

#include <sycl/sycl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "oneapi/dal/algo/hdbscan.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

namespace {

struct blob_spec {
    float cx;
    float cy;
    float sigma;
    std::int64_t count;
};

dal::table make_dataset(const std::vector<blob_spec>& blobs,
                        std::int64_t noise_count,
                        float noise_min,
                        float noise_max,
                        std::uint64_t seed) {
    std::int64_t total = noise_count;
    for (const auto& b : blobs)
        total += b.count;

    auto* raw = new float[total * 2];
    auto data = dal::array<float>(raw, total * 2, [](float* p) {
        delete[] p;
    });

    std::mt19937_64 rng(seed);
    std::int64_t row = 0;
    for (const auto& b : blobs) {
        std::normal_distribution<float> nx(b.cx, b.sigma);
        std::normal_distribution<float> ny(b.cy, b.sigma);
        for (std::int64_t i = 0; i < b.count; ++i) {
            raw[row * 2 + 0] = nx(rng);
            raw[row * 2 + 1] = ny(rng);
            ++row;
        }
    }
    std::uniform_real_distribution<float> ux(noise_min, noise_max);
    std::uniform_real_distribution<float> uy(noise_min, noise_max);
    for (std::int64_t i = 0; i < noise_count; ++i) {
        raw[row * 2 + 0] = ux(rng);
        raw[row * 2 + 1] = uy(rng);
        ++row;
    }
    return dal::homogen_table::wrap(data, total, 2);
}

void print_cluster_summary(const dal::table& responses_tbl, std::int64_t cluster_count) {
    dal::row_accessor<const std::int32_t> acc(responses_tbl);
    auto labels = acc.pull();
    const std::int64_t n = responses_tbl.get_row_count();
    std::vector<std::int64_t> sizes(cluster_count, 0);
    std::int64_t noise = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int32_t lbl = labels[i];
        if (lbl < 0)
            ++noise;
        else if (lbl < cluster_count)
            ++sizes[lbl];
    }
    std::cout << "  noise(-1): " << noise << '\n';
    for (std::int64_t c = 0; c < cluster_count; ++c) {
        std::cout << "  cluster " << c << ": " << sizes[c] << '\n';
    }
}

void run_once(sycl::queue& q,
              const dal::table& data,
              std::int64_t min_cluster_size,
              std::int64_t min_samples,
              bool allow_single_cluster,
              double cluster_selection_epsilon,
              std::int64_t max_cluster_size) {
    auto desc = dal::hdbscan::descriptor<float>(min_cluster_size, min_samples)
                    .set_result_options(dal::hdbscan::result_options::responses)
                    .set_allow_single_cluster(allow_single_cluster)
                    .set_cluster_selection_epsilon(cluster_selection_epsilon)
                    .set_max_cluster_size(max_cluster_size);

    const auto result = dal::compute(q, desc, data);
    const auto cc = result.get_cluster_count();

    std::cout << "min_cluster_size=" << min_cluster_size << " min_samples=" << min_samples
              << " allow_single_cluster=" << (allow_single_cluster ? "true" : "false")
              << " epsilon=" << cluster_selection_epsilon
              << " max_cluster_size=" << max_cluster_size << '\n';
    std::cout << "  cluster_count: " << cc << '\n';
    print_cluster_summary(result.get_responses(), cc);
}

void run(sycl::queue& q) {
    const std::vector<blob_spec> blobs = {
        { /*cx=*/0.0f, /*cy=*/0.0f, /*sigma=*/0.5f, /*count=*/200 },
        { /*cx=*/5.0f, /*cy=*/5.0f, /*sigma=*/0.5f, /*count=*/200 },
        { /*cx=*/0.0f, /*cy=*/5.0f, /*sigma=*/0.5f, /*count=*/200 },
    };
    const std::int64_t noise_count = 60;
    const float noise_min = -2.0f;
    const float noise_max = 7.0f;
    const std::uint64_t seed = 42;

    const auto data = make_dataset(blobs, noise_count, noise_min, noise_max, seed);
    std::cout << "Synthetic data: " << data.get_row_count() << " x " << data.get_column_count()
              << "  (3 Gaussian blobs + uniform noise)\n\n";

    std::cout << "=== Sweep min_samples ===\n";
    for (std::int64_t ms :
         { (std::int64_t)2, (std::int64_t)5, (std::int64_t)10, (std::int64_t)20 }) {
        run_once(q, data, 15, ms, false, 0.0, 0);
    }

    std::cout << "\n=== Sweep min_cluster_size ===\n";
    for (std::int64_t mcs :
         { (std::int64_t)2, (std::int64_t)15, (std::int64_t)100, (std::int64_t)250 }) {
        run_once(q, data, mcs, 5, false, 0.0, 0);
    }

    std::cout << "\n=== Sweep allow_single_cluster ===\n";
    for (bool a : { false, true }) {
        run_once(q, data, 250, 5, a, 0.0, 0);
    }

    std::cout << "\n=== Sweep max_cluster_size ===\n";
    for (std::int64_t mx : { (std::int64_t)0, (std::int64_t)180, (std::int64_t)100 }) {
        run_once(q, data, 15, 5, false, 0.0, mx);
    }

    std::cout << "\n=== Sweep cluster_selection_epsilon ===\n";
    for (double eps : { 0.0, 0.5, 2.0, 5.0 }) {
        run_once(q, data, 15, 5, false, eps, 0);
    }
}

} // namespace

int main(int /*argc*/, char const* /*argv*/[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << '\n'
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
