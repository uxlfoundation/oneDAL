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
#include <limits>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "oneapi/dal/algo/hdbscan/compute.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/algo/hdbscan/test/data.hpp"

namespace oneapi::dal::hdbscan::test {

namespace te = dal::test::engine;

constexpr inline std::uint64_t mask_full = 0xffffffffffffffff;

/// Check that two label arrays define the same partition, up to renumbering.
template <typename Float>
static void check_same_partition(const dal::array<Float>& a_rows,
                                 const dal::array<Float>& b_rows,
                                 std::int64_t row_count) {
    std::map<std::int32_t, std::int32_t> a_to_b;
    std::map<std::int32_t, std::int32_t> b_to_a;

    for (std::int64_t i = 0; i < row_count; i++) {
        const auto a = static_cast<std::int32_t>(a_rows[i]);
        const auto b = static_cast<std::int32_t>(b_rows[i]);

        auto it = a_to_b.find(a);
        if (it == a_to_b.end()) {
            CAPTURE(i, a, b);
            REQUIRE(b_to_a.find(b) == b_to_a.end());
            a_to_b[a] = b;
            b_to_a[b] = a;
        }
        else {
            CAPTURE(i, a, b, it->second);
            REQUIRE(it->second == b);
        }
    }
}

/// Label the points of a single-linkage tree cut at `cut_distance`.
///
/// Port of scikit-learn's `labelling_at_cut`, behind `HDBSCAN.dbscan_clustering`: join every
/// merge below the cut, label each component of at least `min_cluster_size` points, -1 the rest.
///
/// @tparam Float Floating-point type of the dendrogram table
///
/// @param[in] tree             `(n - 1) x 4` dendrogram, `[left, right, distance, size]`
/// @param[in] row_count        Number of original observations `n`
/// @param[in] cut_distance     Merges at or above this distance are not taken
/// @param[in] min_cluster_size Components smaller than this become noise
///
/// @return Per-point labels, length `row_count`
template <typename Float>
static std::vector<std::int32_t> labelling_at_cut(const dal::array<Float>& tree,
                                                  std::int64_t row_count,
                                                  double cut_distance,
                                                  std::int64_t min_cluster_size) {
    const std::int64_t node_count = 2 * row_count - 1;
    REQUIRE(tree.get_count() == 4 * (row_count - 1));

    // Union by rank and path compression as in scikit-learn, since labels follow the root ids.
    std::vector<std::int64_t> parent(node_count);
    std::vector<std::int64_t> rank(node_count, 0);
    for (std::int64_t i = 0; i < node_count; i++) {
        parent[i] = i;
    }

    auto find = [&](std::int64_t x) -> std::int64_t {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](std::int64_t x, std::int64_t y) {
        const std::int64_t xr = find(x);
        const std::int64_t yr = find(y);
        if (xr == yr) {
            return;
        }
        if (rank[xr] < rank[yr]) {
            parent[xr] = yr;
        }
        else if (rank[xr] > rank[yr]) {
            parent[yr] = xr;
        }
        else {
            parent[yr] = xr;
            rank[xr]++;
        }
    };

    for (std::int64_t e = 0; e < row_count - 1; e++) {
        if (static_cast<double>(tree[4 * e + 2]) >= cut_distance) {
            continue;
        }
        const auto node = row_count + e;
        const auto left = static_cast<std::int64_t>(tree[4 * e + 0]);
        const auto right = static_cast<std::int64_t>(tree[4 * e + 1]);
        CAPTURE(e, left, right);
        REQUIRE((0 <= left && left < node));
        REQUIRE((0 <= right && right < node));
        unite(left, node);
        unite(right, node);
    }

    std::map<std::int64_t, std::int64_t> component_size;
    for (std::int64_t i = 0; i < row_count; i++) {
        component_size[find(i)]++;
    }

    std::map<std::int64_t, std::int32_t> component_label;
    std::int32_t next_label = 0;
    for (const auto& [root, size] : component_size) {
        component_label[root] = (size >= min_cluster_size) ? next_label++ : -1;
    }

    std::vector<std::int32_t> labels(row_count);
    for (std::int64_t i = 0; i < row_count; i++) {
        labels[i] = component_label[find(i)];
    }
    return labels;
}

/// Check that two single-linkage trees describe the same hierarchy.
///
/// Merges at an equal distance may come in either order, which renumbers every node above them,
/// so the merge distances and the partition at each of them are compared instead of the rows.
///
/// @tparam Float    Floating-point type of the tree under test
/// @tparam RefFloat Floating-point type of the expected tree
///
/// @param[in] tree      `(n - 1) x 4` dendrogram under test
/// @param[in] ref_tree  `(n - 1) x 4` expected dendrogram
/// @param[in] row_count Number of original observations `n`
/// @param[in] tol       Absolute tolerance on a merge distance
template <typename Float, typename RefFloat>
static void check_same_hierarchy(const dal::array<Float>& tree,
                                 const dal::array<RefFloat>& ref_tree,
                                 std::int64_t row_count,
                                 double tol) {
    const std::int64_t edge_count = row_count - 1;
    REQUIRE(tree.get_count() == 4 * edge_count);
    REQUIRE(ref_tree.get_count() == 4 * edge_count);

    INFO("merge distances agree");
    for (std::int64_t e = 0; e < edge_count; e++) {
        CAPTURE(e, tree[4 * e + 2], ref_tree[4 * e + 2]);
        REQUIRE(std::abs(double(tree[4 * e + 2]) - double(ref_tree[4 * e + 2])) < tol);
    }

    // Each component is named after its smallest member, so equal partitions compare equal.
    const auto canonical = [=](const std::vector<std::int32_t>& labels) {
        std::vector<std::int32_t> first_point;
        std::map<std::int32_t, std::int32_t> label_to_first;
        for (std::int64_t i = 0; i < row_count; i++) {
            const auto it = label_to_first.emplace(labels[i], static_cast<std::int32_t>(i));
            first_point.push_back(it.first->second);
        }
        return first_point;
    };

    INFO("the induced partition agrees at every merge distance");
    // At most this many levels are cut, evenly spread, so large trees stay cheap to check.
    constexpr std::int64_t max_levels = 256;
    const std::int64_t stride = std::max<std::int64_t>(1, edge_count / max_levels);
    for (std::int64_t e = 0; e < edge_count; e++) {
        // Cut halfway to the next merge distance; merges within `2 * tol` of each other are only
        // checked together, at the last of them.
        const double distance = static_cast<double>(ref_tree[4 * e + 2]);
        const bool has_next = (e + 1 < edge_count);
        const double next = has_next ? static_cast<double>(ref_tree[4 * (e + 1) + 2]) : 0.0;
        if ((has_next && next - distance <= 2 * tol) || (has_next && e % stride != 0)) {
            continue;
        }
        const double cut = has_next ? 0.5 * (distance + next) : (distance + 1.0);
        CAPTURE(e, distance, cut);

        const auto actual = canonical(labelling_at_cut(tree, row_count, cut, 1));
        const auto expected = canonical(labelling_at_cut(ref_tree, row_count, cut, 1));
        const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        if (mismatch.first != actual.end()) {
            const auto i = std::distance(actual.begin(), mismatch.first);
            CAPTURE(i, actual[i], expected[i]);
            REQUIRE(actual[i] == expected[i]);
        }
    }
}

/// Check that every merge away from a tie has the same left and right child as the expected tree.
///
/// A child is named after its smallest point, which does not depend on how tied merges below it
/// were ordered.
///
/// @tparam Float    Floating-point type of the tree under test
/// @tparam RefFloat Floating-point type of the expected tree
///
/// @param[in] tree      `(n - 1) x 4` dendrogram under test
/// @param[in] ref_tree  `(n - 1) x 4` expected dendrogram
/// @param[in] row_count Number of original observations `n`
/// @param[in] tol       Merges whose distances differ by at most `2 * tol` count as tied
template <typename Float, typename RefFloat>
static void check_same_children(const dal::array<Float>& tree,
                                const dal::array<RefFloat>& ref_tree,
                                std::int64_t row_count,
                                double tol) {
    const std::int64_t edge_count = row_count - 1;
    REQUIRE(tree.get_count() == 4 * edge_count);
    REQUIRE(ref_tree.get_count() == 4 * edge_count);

    const auto smallest_point = [=](const auto& t) {
        std::vector<std::int64_t> smallest(row_count + edge_count);
        for (std::int64_t i = 0; i < row_count; i++) {
            smallest[i] = i;
        }
        for (std::int64_t e = 0; e < edge_count; e++) {
            const auto left = static_cast<std::int64_t>(t[4 * e + 0]);
            const auto right = static_cast<std::int64_t>(t[4 * e + 1]);
            CAPTURE(e, left, right);
            REQUIRE((0 <= left && left < row_count + e));
            REQUIRE((0 <= right && right < row_count + e));
            smallest[row_count + e] = std::min(smallest[left], smallest[right]);
        }
        return smallest;
    };
    const auto actual = smallest_point(tree);
    const auto expected = smallest_point(ref_tree);

    const auto distance = [&](std::int64_t e) {
        return static_cast<double>(ref_tree[4 * e + 2]);
    };
    for (std::int64_t e = 0; e < edge_count; e++) {
        if ((e > 0 && distance(e) - distance(e - 1) <= 2 * tol) ||
            (e + 1 < edge_count && distance(e + 1) - distance(e) <= 2 * tol)) {
            continue;
        }
        CAPTURE(e, distance(e));
        for (std::int64_t side = 0; side < 2; side++) {
            CAPTURE(side);
            REQUIRE(actual[static_cast<std::int64_t>(tree[4 * e + side])] ==
                    expected[static_cast<std::int64_t>(ref_tree[4 * e + side])]);
        }
        REQUIRE(static_cast<std::int64_t>(tree[4 * e + 3]) ==
                static_cast<std::int64_t>(ref_tree[4 * e + 3]));
    }
}

template <typename TestType, typename Derived>
class hdbscan_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using Float = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using result_t = compute_result<task::clustering>;
    using input_t = compute_input<task::clustering>;

    auto get_descriptor(std::int64_t min_cluster_size, std::int64_t min_samples) const {
        return hdbscan::descriptor<Float, method_t>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses);
    }

    auto get_descriptor(std::int64_t min_cluster_size,
                        std::int64_t min_samples,
                        distance_metric metric,
                        double degree = 2.0) const {
        return get_method_descriptor<method_t>(min_cluster_size, min_samples, metric, degree);
    }

    auto get_descriptor(std::int64_t min_cluster_size,
                        std::int64_t min_samples,
                        const result_option_id& options) const {
        return hdbscan::descriptor<Float, method_t>(min_cluster_size, min_samples)
            .set_result_options(options);
    }

    /// Descriptor of `Method` that computes the responses, for cross-method comparisons.
    template <typename Method>
    static auto get_method_descriptor(std::int64_t min_cluster_size,
                                      std::int64_t min_samples,
                                      distance_metric metric = distance_metric::euclidean,
                                      double degree = 2.0) {
        return hdbscan::descriptor<Float, Method>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses)
            .set_metric(metric)
            .set_degree(degree);
    }

    static dal::array<Float> get_labels(const result_t& result) {
        return row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
    }

    static dal::array<Float> get_probabilities(const result_t& result) {
        return row_accessor<const Float>(result.get_probabilities()).pull({ 0, -1 });
    }

    static dal::array<Float> get_single_linkage_tree(const result_t& result) {
        return row_accessor<const Float>(result.get_single_linkage_tree()).pull({ 0, -1 });
    }

    /// Number of points labeled noise.
    static std::int64_t count_noise(const dal::array<Float>& labels) {
        std::int64_t count = 0;
        for (std::int64_t i = 0; i < labels.get_count(); i++) {
            count += (static_cast<std::int32_t>(labels[i]) < 0);
        }
        return count;
    }

    /// Size of every reported (non-noise) cluster, keyed by label.
    static std::map<std::int64_t, std::int64_t> cluster_sizes(const dal::array<Float>& labels) {
        std::map<std::int64_t, std::int64_t> sizes;
        for (std::int64_t i = 0; i < labels.get_count(); i++) {
            const auto label = static_cast<std::int64_t>(labels[i]);
            if (label >= 0) {
                sizes[label]++;
            }
        }
        return sizes;
    }

    /// Check `actual[i]` against `expected[i]` within the absolute tolerance `tol`.
    ///
    /// `te::check_if_tables_equal_approx` compares relative errors, which is too strict for the
    /// small membership strengths these references hold.
    static void check_close(const dal::array<Float>& actual,
                            const double* expected,
                            std::int64_t count,
                            double tol) {
        REQUIRE(actual.get_count() == count);
        for (std::int64_t i = 0; i < count; i++) {
            CAPTURE(i, actual[i], expected[i]);
            REQUIRE(std::abs(double(actual[i]) - expected[i]) < tol);
        }
    }

    static void check_close(const dal::array<Float>& actual,
                            const dal::array<Float>& expected,
                            double tol) {
        REQUIRE(actual.get_count() == expected.get_count());
        for (std::int64_t i = 0; i < actual.get_count(); i++) {
            CAPTURE(i, actual[i], expected[i]);
            REQUIRE(std::abs(double(actual[i]) - double(expected[i])) < tol);
        }
    }

    /// Check that two results have the same cluster count and the same partition.
    static void check_same_result(const result_t& a, const result_t& b) {
        REQUIRE(a.get_cluster_count() == b.get_cluster_count());
        const auto a_labels = get_labels(a);
        const auto b_labels = get_labels(b);
        REQUIRE(a_labels.get_count() == b_labels.get_count());
        check_same_partition(a_labels, b_labels, a_labels.get_count());
    }

    /// Run two descriptors on `data` and check that they give the same partition.
    template <typename DescA, typename DescB>
    void check_descriptors_agree(const DescA& a, const DescB& b, const table& data) {
        check_same_result(this->compute(a, data), this->compute(b, data));
    }

    /// Check that each of `Methods` gives the partition of the method under test.
    template <typename... Methods>
    void check_methods_agree(const table& data,
                             std::int64_t min_cluster_size,
                             std::int64_t min_samples,
                             distance_metric metric = distance_metric::euclidean,
                             double degree = 2.0) {
        const auto reference = this->compute(
            get_method_descriptor<method_t>(min_cluster_size, min_samples, metric, degree),
            data);
        (check_same_result(
             reference,
             this->compute(
                 get_method_descriptor<Methods>(min_cluster_size, min_samples, metric, degree),
                 data)),
         ...);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Check that `desc` on the device gives the partition of the same method in float64 on the
    /// host, which tells float32 rounding apart from a wrong neighbor.
    template <typename Desc>
    std::pair<result_t, result_t> compare_device_with_float64_host(const Desc& desc,
                                                                   const table& data) {
        const auto reference_desc =
            hdbscan::descriptor<double, method_t>(desc.get_min_cluster_size(),
                                                  desc.get_min_samples())
                .set_metric(desc.get_metric())
                .set_degree(desc.get_degree())
                .set_alpha(desc.get_alpha())
                .set_result_options(desc.get_result_options());
        const auto host = dal::compute(reference_desc, data);
        const auto device = dal::compute(this->get_policy().get_queue(), desc, data);
        check_same_result(host, device);
        return { host, device };
    }

    /// Run `desc` on the host and on the device of the test policy, and check that the two
    /// partitions agree when the responses were requested.
    ///
    /// @return The host and the device results
    template <typename Desc>
    std::pair<result_t, result_t> compare_host_and_device(const Desc& desc, const table& data) {
        const auto host = dal::compute(desc, data);
        const auto device = dal::compute(this->get_policy().get_queue(), desc, data);
        if (desc.get_result_options().test(result_options::responses)) {
            check_same_result(host, device);
        }
        return { host, device };
    }
#endif

    void run_checks(const table& data,
                    std::int64_t min_cluster_size,
                    std::int64_t min_samples,
                    std::int64_t expected_cluster_count) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples);

        INFO("run compute");
        const auto compute_result = this->compute(hdbscan_desc, data);

        check_compute_result(compute_result, data, expected_cluster_count);
    }

    void run_checks(const table& data,
                    std::int64_t min_cluster_size,
                    std::int64_t min_samples,
                    distance_metric metric,
                    double degree,
                    std::int64_t expected_cluster_count) {
        CAPTURE(min_cluster_size, min_samples, static_cast<int>(metric), degree);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples, metric, degree);

        INFO("run compute");
        const auto compute_result = this->compute(hdbscan_desc, data);

        check_compute_result(compute_result, data, expected_cluster_count);
    }

    void check_compute_result(const result_t& compute_result,
                              const table& data,
                              std::int64_t expected_cluster_count) {
        INFO("check cluster count");
        REQUIRE(compute_result.get_cluster_count() >= 0);

        if (expected_cluster_count >= 0) {
            REQUIRE(compute_result.get_cluster_count() == expected_cluster_count);
        }

        INFO("check responses shape");
        const auto responses = compute_result.get_responses();
        REQUIRE(responses.get_row_count() == data.get_row_count());
        REQUIRE(responses.get_column_count() == 1);

        INFO("check response values");
        const auto rows = row_accessor<const Float>(responses).pull({ 0, -1 });
        for (std::int64_t i = 0; i < data.get_row_count(); i++) {
            const auto label = static_cast<std::int32_t>(rows[i]);
            // Labels should be >= -1 (noise) and < cluster_count
            REQUIRE(label >= -1);
            if (compute_result.get_cluster_count() > 0) {
                REQUIRE(label < compute_result.get_cluster_count());
            }
        }
    }

    void run_checks_with_responses(const table& data,
                                   std::int64_t min_cluster_size,
                                   std::int64_t min_samples,
                                   const table& ref_responses) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples);

        INFO("run compute");
        const auto compute_result = this->compute(hdbscan_desc, data);

        INFO("check responses match reference");
        check_responses_against_ref(compute_result.get_responses(), ref_responses);
    }

    void check_responses_against_ref(const table& responses, const table& ref_responses) {
        ONEDAL_ASSERT(responses.get_row_count() == ref_responses.get_row_count());
        ONEDAL_ASSERT(responses.get_column_count() == ref_responses.get_column_count());
        ONEDAL_ASSERT(responses.get_column_count() == 1);
        const auto row_count = responses.get_row_count();
        const auto rows = row_accessor<const Float>(responses).pull({ 0, -1 });
        const auto ref_rows = row_accessor<const Float>(ref_responses).pull({ 0, -1 });
        for (std::int64_t i = 0; i < row_count; i++) {
            CAPTURE(i, rows[i], ref_rows[i]);
            REQUIRE(ref_rows[i] == rows[i]);
        }
    }

    /// Check the responses against reference labels entry by entry.
    void check_responses_against_ref(const table& responses,
                                     const std::int32_t* ref_labels,
                                     std::int64_t row_count) {
        check_responses_against_ref(responses, homogen_table::wrap(ref_labels, row_count, 1));
    }

    void mode_checks(result_option_id compute_mode,
                     const table& data,
                     std::int64_t min_cluster_size,
                     std::int64_t min_samples) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc =
            get_descriptor(min_cluster_size, min_samples).set_result_options(compute_mode);

        INFO("run compute");
        const auto compute_result = this->compute(hdbscan_desc, data);

        INFO("check mode");
        check_for_exception_for_non_requested_results(compute_mode, compute_result);

        if (compute_mode.test(result_options::probabilities)) {
            INFO("check requested probabilities are populated");
            check_probabilities(compute_result, data.get_row_count());
        }

        if (compute_mode.test(result_options::single_linkage_tree)) {
            INFO("check requested single linkage tree is populated");
            check_single_linkage_tree(compute_result, data.get_row_count());
        }
    }

    /// Check the shape and the invariants of any single-linkage tree: ascending distances,
    /// children formed before their merge, sizes that add up, every node merged once and a root
    /// over all points.
    ///
    /// @param[in] result    Compute result with `single_linkage_tree` requested
    /// @param[in] row_count Number of input observations
    void check_single_linkage_tree(const result_t& result, std::int64_t row_count) {
        const auto tree_table = result.get_single_linkage_tree();
        const std::int64_t edge_count = row_count - 1;
        REQUIRE(tree_table.get_row_count() == edge_count);
        REQUIRE(tree_table.get_column_count() == 4);

        const auto tree = row_accessor<const Float>(tree_table).pull({ 0, -1 });

        std::vector<std::int64_t> node_size(row_count + edge_count, 1);
        // Number of later merges that take each node.
        std::vector<std::int32_t> consumed(row_count + edge_count, 0);

        Float previous = -std::numeric_limits<Float>::infinity();
        for (std::int64_t e = 0; e < edge_count; e++) {
            const auto left = static_cast<std::int64_t>(tree[4 * e + 0]);
            const auto right = static_cast<std::int64_t>(tree[4 * e + 1]);
            const Float distance = tree[4 * e + 2];
            const auto size = static_cast<std::int64_t>(tree[4 * e + 3]);
            CAPTURE(e, left, right, distance, size);

            REQUIRE(distance >= Float(0));
            REQUIRE(distance >= previous);
            previous = distance;

            // Row `e` forms node `row_count + e`, so its children have smaller ids.
            REQUIRE(left >= 0);
            REQUIRE(right >= 0);
            REQUIRE(left < row_count + e);
            REQUIRE(right < row_count + e);
            REQUIRE(left != right);

            REQUIRE(size == node_size[left] + node_size[right]);
            node_size[row_count + e] = size;
            consumed[left]++;
            consumed[right]++;
        }

        const std::int64_t root = row_count + edge_count - 1;
        REQUIRE(node_size[root] == row_count);
        for (std::int64_t n = 0; n < row_count + edge_count; n++) {
            CAPTURE(n, consumed[n]);
            REQUIRE(consumed[n] == ((n == root) ? 0 : 1));
        }
    }

    /// Run compute with `responses | single_linkage_tree` and check both outputs.
    ///
    /// @param[in] data             Input data table
    /// @param[in] min_cluster_size `min_cluster_size` parameter value
    /// @param[in] min_samples      `min_samples` parameter value
    void run_single_linkage_tree_checks(const table& data,
                                        std::int64_t min_cluster_size,
                                        std::int64_t min_samples) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples)
                                      .set_result_options(result_options::responses |
                                                          result_options::single_linkage_tree);

        INFO("run compute");
        const auto compute_result = this->compute(hdbscan_desc, data);

        INFO("check single linkage tree");
        check_single_linkage_tree(compute_result, data.get_row_count());
    }

    /// Check the shape and the invariants of the probabilities: all in `[0, 1]`, exactly 0 for
    /// noise, positive otherwise and 1 for the most persistent point. The labels are only checked
    /// when the responses were requested.
    ///
    /// @param[in] result    Compute result with `probabilities` requested
    /// @param[in] row_count Number of input observations
    void check_probabilities(const result_t& result, std::int64_t row_count) {
        const auto probabilities = result.get_probabilities();
        REQUIRE(probabilities.get_row_count() == row_count);
        REQUIRE(probabilities.get_column_count() == 1);

        const auto probs = row_accessor<const Float>(probabilities).pull({ 0, -1 });

        const bool has_responses = result.get_result_options().test(result_options::responses);
        dal::array<Float> labels;
        if (has_responses) {
            labels = row_accessor<const Float>(result.get_responses()).pull({ 0, -1 });
        }

        Float max_prob = Float(0);
        for (std::int64_t i = 0; i < row_count; i++) {
            CAPTURE(i, probs[i]);
            REQUIRE(probs[i] >= Float(0));
            REQUIRE(probs[i] <= Float(1));
            if (has_responses) {
                const auto label = static_cast<std::int32_t>(labels[i]);
                if (label < 0) {
                    REQUIRE(probs[i] == Float(0));
                }
                else {
                    REQUIRE(probs[i] > Float(0));
                    max_prob = std::max(max_prob, probs[i]);
                }
            }
        }

        if (has_responses && result.get_cluster_count() > 0) {
            INFO("the most persistent clustered point reaches full membership");
            REQUIRE(max_prob == Float(1));
        }
    }

    /// Run compute with `responses | probabilities` and check both outputs together.
    ///
    /// @param[in] data             Input data table
    /// @param[in] min_cluster_size `min_cluster_size` parameter value
    /// @param[in] min_samples      `min_samples` parameter value
    void run_probability_checks(const table& data,
                                std::int64_t min_cluster_size,
                                std::int64_t min_samples) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc =
            get_descriptor(min_cluster_size, min_samples)
                .set_result_options(result_options::responses | result_options::probabilities);

        INFO("run compute");
        const auto compute_result = this->compute(hdbscan_desc, data);

        INFO("check probabilities");
        check_probabilities(compute_result, data.get_row_count());
    }

    void check_for_exception_for_non_requested_results(result_option_id compute_mode,
                                                       const result_t& result) {
        if (!compute_mode.test(result_options::responses)) {
            REQUIRE_THROWS_AS(result.get_responses(), domain_error);
        }
        if (!compute_mode.test(result_options::core_flags)) {
            REQUIRE_THROWS_AS(result.get_core_flags(), domain_error);
        }
        if (!compute_mode.test(result_options::core_observations)) {
            REQUIRE_THROWS_AS(result.get_core_observations(), domain_error);
        }
        if (!compute_mode.test(result_options::core_observation_indices)) {
            REQUIRE_THROWS_AS(result.get_core_observation_indices(), domain_error);
        }
        if (!compute_mode.test(result_options::cluster_centers)) {
            REQUIRE_THROWS_AS(result.get_cluster_centers(), domain_error);
        }
        if (!compute_mode.test(result_options::medoid_centers)) {
            REQUIRE_THROWS_AS(result.get_medoid_centers(), domain_error);
        }
        if (!compute_mode.test(result_options::probabilities)) {
            REQUIRE_THROWS_AS(result.get_probabilities(), domain_error);
        }
        if (!compute_mode.test(result_options::single_linkage_tree)) {
            REQUIRE_THROWS_AS(result.get_single_linkage_tree(), domain_error);
        }
    }
};

} // namespace oneapi::dal::hdbscan::test
