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

/// Check that two label arrays define the same partition (permutation-invariant).
/// Different implementations may assign different integer labels to the same clusters
/// because floating-point differences can reorder MST edges.
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
/// Transcription of scikit-learn's `labelling_at_cut`, the routine behind
/// `HDBSCAN.dbscan_clustering`: union every merge below the cut, then hand a
/// dense label to each component holding at least `min_cluster_size` points and
/// `-1` to the rest. The test uses it to check that the exported dendrogram is
/// usable for exactly that purpose, so it is written against the table only --
/// it never looks at the data or the flat responses.
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

    // Union-find over the whole dendrogram, points and merge nodes alike, with the
    // union by rank and the find by path compression the reference uses. The two
    // match because the label of a component is handed out in the order of its root
    // id, and the root id depends on how the unions were taken.
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
        // Both children join the node the merge created, so a chain of merges below
        // the cut collapses into one component.
        const auto node = row_count + e;
        unite(static_cast<std::int64_t>(tree[4 * e + 0]), node);
        unite(static_cast<std::int64_t>(tree[4 * e + 1]), node);
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
/// The rows are deliberately not compared one by one. Merges at an equal distance
/// may be taken in either order, which shifts the node ids of everything above
/// them without changing the hierarchy, and that does happen between two
/// implementations and even between two backends of this one. What is well defined
/// is the sequence of merge distances and the partition the tree induces at each of
/// them, which is also all a re-cut of the hierarchy can observe.
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

    INFO("merge distances agree");
    for (std::int64_t e = 0; e < edge_count; e++) {
        CAPTURE(e, tree[4 * e + 2], ref_tree[4 * e + 2]);
        REQUIRE(std::abs(double(tree[4 * e + 2]) - double(ref_tree[4 * e + 2])) < tol);
    }

    // Component id of each point, named after its smallest member, so two
    // labellings of the same partition compare equal whatever order they numbered
    // the components in.
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
    for (std::int64_t e = 0; e < edge_count; e++) {
        // Halfway to the next distinct merge distance, so merge `e` is taken and
        // merge `e + 1` is not, with room to spare for the gap between the two
        // trees. A tie has no level of its own: the pair of merges is only
        // observable together, at the level of the second one.
        const double distance = static_cast<double>(ref_tree[4 * e + 2]);
        const bool has_next = (e + 1 < edge_count);
        const double next = has_next ? static_cast<double>(ref_tree[4 * (e + 1) + 2]) : 0.0;
        if (has_next && next == distance) {
            continue;
        }
        const double cut = has_next ? 0.5 * (distance + next) : (distance + 1.0);
        CAPTURE(e, distance, cut);

        const auto actual = canonical(labelling_at_cut(tree, row_count, cut, 1));
        const auto expected = canonical(labelling_at_cut(ref_tree, row_count, cut, 1));
        for (std::int64_t i = 0; i < row_count; i++) {
            CAPTURE(i, actual[i], expected[i]);
            REQUIRE(actual[i] == expected[i]);
        }
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
        return hdbscan::descriptor<Float, method_t>(min_cluster_size, min_samples)
            .set_result_options(result_options::responses)
            .set_metric(metric)
            .set_degree(degree);
    }

    void run_checks(const table& data,
                    std::int64_t min_cluster_size,
                    std::int64_t min_samples,
                    std::int64_t expected_cluster_count) {
        CAPTURE(min_cluster_size, min_samples);

        INFO("create descriptor");
        const auto hdbscan_desc = get_descriptor(min_cluster_size, min_samples);

        INFO("run compute");
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

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
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

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
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

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
            REQUIRE(ref_rows[i] == rows[i]);
        }
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
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

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

    /// Check the shape and the structural invariants of a `single_linkage_tree` table.
    ///
    /// Everything checked here holds for any valid single-linkage dendrogram, so the
    /// check applies to every input: the documented shape, merge distances ascending
    /// (the hierarchy is built from an ascending sweep of the MST, and a re-cut is
    /// only meaningful if they are monotone), ids referring only to already-formed
    /// nodes, subtree sizes equal to the sum of the two children, every node consumed
    /// by exactly one later merge, and a root spanning all `row_count` points.
    ///
    /// @param[in] result    Compute result with `single_linkage_tree` requested
    /// @param[in] row_count Number of input observations
    void check_single_linkage_tree(const result_t& result, std::int64_t row_count) {
        const auto tree_table = result.get_single_linkage_tree();
        const std::int64_t edge_count = row_count - 1;
        REQUIRE(tree_table.get_row_count() == edge_count);
        REQUIRE(tree_table.get_column_count() == 4);

        const auto tree = row_accessor<const Float>(tree_table).pull({ 0, -1 });

        // Size of every node, leaves included, so the per-merge sum can be checked.
        std::vector<std::int64_t> node_size(row_count + edge_count, 1);
        // How many later merges consumed each node. A dendrogram is a tree, so the
        // root is consumed zero times and every other node exactly once.
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

            // `row_count + e` is this row's own node, so both children must be
            // strictly below it: a merge can only consume what already exists.
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
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

        INFO("check single linkage tree");
        check_single_linkage_tree(compute_result, data.get_row_count());
    }

    /// Check the shape and the value invariants of a requested `probabilities` table.
    ///
    /// Only the invariants that hold for every input are checked: the table has the
    /// documented shape, every membership strength lies in `[0, 1]`, a noise point
    /// gets exactly 0 and a clustered point gets a strictly positive value, and the
    /// most persistent point of the whole dataset reaches 1 (the deepest surviving
    /// point of a cluster drops out at that cluster's death lambda, which is what the
    /// ratio normalizes by). The responses are only cross-checked when they were
    /// requested too, so this also covers the probabilities-only mode.
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
        const auto compute_result =
            oneapi::dal::test::engine::compute(this->get_policy(), hdbscan_desc, data);

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
