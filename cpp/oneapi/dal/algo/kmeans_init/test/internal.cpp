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

#include <algorithm>
#include <vector>

#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

#include "src/algorithms/kmeans/kmeans_init_kernel.h"
#include "src/algorithms/kmeans/kmeans_plusplus_init_impl.i"

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;
namespace interop = dal::backend::interop;
namespace daal_init = daal::algorithms::kmeans::init::internal;

constexpr auto daal_cpu = daal::internal::firstCpuType;

// Each trial's slot must be scored with that trial's own candidate, the previous best trial's
// slot included, and every slot must start from the previous best trial's distances.
TEMPLATE_TEST("kmeans++ updateMinDistInBlock scores each trial with its own candidate",
              "[kmeans_init][internal][plus_plus]",
              float,
              double) {
    using float_t = TestType;
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 2;
    constexpr std::size_t trial_count = 3;

    const std::vector<float_t> data = { 0, 0, 3, 0, 0, 4, 6, 8 };
    const auto dense_table = interop::copy_to_daal_homogen_table<float_t>(
        homogen_table::wrap(data.data(), row_count, column_count));
    const auto csr_table = interop::copy_to_daal_csr_table<float_t>(
        te::dense_to_explicit_csr(data.data(),
                                  row_count,
                                  column_count,
                                  sparse_indexing::one_based));
    daal_init::DataHelperDense<float_t, daal_cpu> dense(dense_table.get());
    daal_init::DataHelperCSR<float_t, daal_cpu> csr(csr_table.get());

    // Trial t's candidate is row t.
    const std::vector<float_t> candidates(data.begin(), data.begin() + trial_count * column_count);
    const std::vector<float_t> previous_min_dist = { 1000, 1000, 1000, 50 };

    // min(previous_min_dist, squared distance to row t) for each trial t and row.
    const float_t expected_dist[trial_count][row_count] = { { 0, 9, 16, 50 },
                                                            { 9, 0, 25, 50 },
                                                            { 16, 25, 0, 50 } };
    const float_t expected_potential[trial_count] = { 75, 84, 91 };

    auto check = [&](auto& data_helper) {
        for (std::size_t best_trial = 0; best_trial < trial_count; ++best_trial) {
            CAPTURE(best_trial);
            std::vector<float_t> min_dist(trial_count * row_count, float_t(-1));
            std::copy(previous_min_dist.begin(),
                      previous_min_dist.end(),
                      min_dist.begin() + best_trial * row_count);
            std::vector<float_t> potential(trial_count, float_t(-1));

            const auto status = data_helper.updateMinDistInBlock(potential.data(),
                                                                 1,
                                                                 0,
                                                                 trial_count,
                                                                 best_trial,
                                                                 nullptr,
                                                                 candidates.data(),
                                                                 min_dist.data());
            REQUIRE(status.ok());

            for (std::size_t t = 0; t < trial_count; ++t) {
                CAPTURE(t);
                REQUIRE(potential[t] == expected_potential[t]);
                for (std::int64_t r = 0; r < row_count; ++r) {
                    CAPTURE(r);
                    REQUIRE(min_dist[t * row_count + r] == expected_dist[t][r]);
                }
            }
        }
    };
    check(dense);
    check(csr);
}

} // namespace oneapi::dal::kmeans_init::test
