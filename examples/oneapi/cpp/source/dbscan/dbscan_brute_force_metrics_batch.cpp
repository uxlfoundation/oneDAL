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

#include "oneapi/dal/algo/dbscan.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

// epsilon is in the units of the metric, so each metric gets its own value.
struct metric_case {
    const char *name;
    dal::dbscan::distance_metric metric;
    double epsilon;
};

const metric_case metric_cases[] = {
    { "euclidean", dal::dbscan::distance_metric::euclidean, 0.3 },
    { "manhattan", dal::dbscan::distance_metric::manhattan, 0.4 },
    { "minkowski (degree 3)", dal::dbscan::distance_metric::minkowski, 0.3 },
    { "chebyshev", dal::dbscan::distance_metric::chebyshev, 0.25 },
    { "cosine", dal::dbscan::distance_metric::cosine, 0.001 },
};

int main(int argc, char const *argv[]) {
    const auto data_file_name = get_data_path("data/hdbscan_dense.csv");

    const auto x_data = dal::read<dal::table>(dal::csv::data_source{ data_file_name });

    const std::int64_t min_observations = 10;

    for (const auto &c : metric_cases) {
        auto dbscan_desc = dal::dbscan::descriptor<>(c.epsilon, min_observations)
                               .set_metric(c.metric)
                               .set_degree(3.0)
                               .set_result_options(dal::dbscan::result_options::responses);

        const auto result_compute = dal::compute(dbscan_desc, x_data);

        std::cout << "Metric: " << c.name << ", epsilon: " << c.epsilon
                  << ", cluster count: " << result_compute.get_cluster_count() << std::endl;
    }
    return 0;
}
