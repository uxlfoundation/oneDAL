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

// HDBSCAN tunable playground.
//
// Generates a synthetic 2-D dataset (three Gaussian blobs of configurable
// size / spread / center, plus optional uniform noise) and runs HDBSCAN
// with parameters listed at the top of `main`. Prints the resulting cluster
// count and the size of every cluster so changes in input or parameters
// translate directly into observable changes in output.
//
// Recommended sweeps:
//   * min_samples in {2, 5, 10, 20}            -> tighter cores merge
//                                                 / split borderline pts
//   * min_cluster_size in {2, 5, 15, 50}       -> small blobs get absorbed
//                                                 into noise (-1)
//   * allow_single_cluster in {false, true}    -> with overlapping blobs and
//                                                 large min_cluster_size,
//                                                 setting true collapses to
//                                                 one cluster
//   * max_cluster_size > 0                     -> oversized clusters are
//                                                 broken apart
//   * cluster_selection_epsilon > 0            -> nearby clusters merge

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "oneapi/dal/algo/hdbscan.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

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

void run_once(const dal::table& data,
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

    const auto result = dal::compute(desc, data);
    const auto cc = result.get_cluster_count();

    std::cout << "min_cluster_size=" << min_cluster_size << " min_samples=" << min_samples
              << " allow_single_cluster=" << (allow_single_cluster ? "true" : "false")
              << " epsilon=" << cluster_selection_epsilon
              << " max_cluster_size=" << max_cluster_size << '\n';
    std::cout << "  cluster_count: " << cc << '\n';
    print_cluster_summary(result.get_responses(), cc);
}

} // namespace

int main(int /*argc*/, char const* /*argv*/[]) {
    // ---- Edit these to change the dataset ------------------------------
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

    // ---- Sweep #1: min_samples (small=permissive cores, large=strict) ---
    std::cout << "=== Sweep min_samples ===\n";
    for (std::int64_t ms :
         { (std::int64_t)2, (std::int64_t)5, (std::int64_t)10, (std::int64_t)20 }) {
        run_once(data,
                 /*mcs=*/15,
                 /*ms=*/ms,
                 /*allow_single=*/false,
                 /*epsilon=*/0.0,
                 /*max_cs=*/0);
    }

    // ---- Sweep #2: min_cluster_size (large -> small blobs become noise) -
    std::cout << "\n=== Sweep min_cluster_size ===\n";
    for (std::int64_t mcs :
         { (std::int64_t)2, (std::int64_t)15, (std::int64_t)100, (std::int64_t)250 }) {
        run_once(data,
                 /*mcs=*/mcs,
                 /*ms=*/5,
                 /*allow_single=*/false,
                 /*epsilon=*/0.0,
                 /*max_cs=*/0);
    }

    // ---- Sweep #3: allow_single_cluster --------------------------------
    // With a min_cluster_size that exceeds any single blob, EOM normally
    // wants the root. allow_single_cluster=false should now produce 0 / few
    // clusters (no rescue selecting direct children of the root any more);
    // setting it to true returns a single cluster that engulfs the blobs.
    std::cout << "\n=== Sweep allow_single_cluster ===\n";
    for (bool a : { false, true }) {
        run_once(data,
                 /*mcs=*/250,
                 /*ms=*/5,
                 /*allow_single=*/a,
                 /*epsilon=*/0.0,
                 /*max_cs=*/0);
    }

    // ---- Sweep #4: max_cluster_size ------------------------------------
    // Setting max_cluster_size below a blob's size forces EOM to deselect
    // it and prefer its children (or fall back to noise if no children).
    std::cout << "\n=== Sweep max_cluster_size ===\n";
    for (std::int64_t mx : { (std::int64_t)0, (std::int64_t)180, (std::int64_t)100 }) {
        run_once(data,
                 /*mcs=*/15,
                 /*ms=*/5,
                 /*allow_single=*/false,
                 /*epsilon=*/0.0,
                 /*max_cs=*/mx);
    }

    // ---- Sweep #5: cluster_selection_epsilon ----------------------------
    std::cout << "\n=== Sweep cluster_selection_epsilon ===\n";
    for (double eps : { 0.0, 0.5, 2.0, 5.0 }) {
        run_once(data,
                 /*mcs=*/15,
                 /*ms=*/5,
                 /*allow_single=*/false,
                 /*epsilon=*/eps,
                 /*max_cs=*/0);
    }

    return 0;
}
