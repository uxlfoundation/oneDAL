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

#include "oneapi/dal/algo/decision_forest/test/fixture.hpp"

namespace oneapi::dal::decision_forest::test {

namespace te = dal::test::engine;

template <typename TestType>
class df_best_first_test : public df_test<TestType, df_best_first_test<TestType>> {
public:
    using base_t = df_test<TestType, df_best_first_test<TestType>>;
    using float_t = typename base_t::float_t;

    // Trains a single unbootstrapped tree with best-first growth (max_leaf_nodes)
    // on inline x/y arrays and returns the predicted response for every training
    // point, in order.
    std::vector<double> train_and_predict_all(const float* x,
                                               const float* y,
                                               std::int64_t n,
                                               std::int64_t max_leaf_nodes,
                                               double min_impurity_decrease = 0.0) {
        const te::dataframe x_df{ array<float>::wrap(x, n), n, 1 };
        const te::dataframe y_df{ array<float>::wrap(y, n), n, 1 };

        auto desc = this->get_default_descriptor();
        desc.set_tree_count(1);
        desc.set_bootstrap(false);
        desc.set_features_per_node(1);
        desc.set_min_observations_in_leaf_node(1);
        desc.set_min_observations_in_split_node(2);
        desc.set_max_bins(n);
        desc.set_min_bin_size(1);
        desc.set_max_leaf_nodes(max_leaf_nodes);
        desc.set_min_impurity_decrease_in_split_node(min_impurity_decrease);

        const auto x_table = x_df.get_table(this->get_homogen_table_id(), range(0, -1));
        const auto y_table = y_df.get_table(this->get_homogen_table_id(), range(0, -1));

        const auto train_result = this->train(desc, x_table, y_table);
        const auto infer_result = this->infer(desc, train_result.get_model(), x_table);

        const auto responses = dal::row_accessor<const float_t>(infer_result.get_responses()).pull();
        std::vector<double> result(n);
        for (std::int64_t i = 0; i < n; i++) {
            result[i] = static_cast<double>(responses[i]);
        }
        return result;
    }
};

using df_best_first_types = _TE_COMBINE_TYPES_3((float, double),
                                                (df::method::dense, df::method::hist),
                                                (df::task::regression));

#define DF_BEST_FIRST_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_best_first_test, name, "[df][integration][best-first]", df_best_first_types)

// Regression test for https://github.com/uxlfoundation/oneDAL/issues/3771 /
// https://github.com/uxlfoundation/oneDAL/pull/3772.
//
// buildNode's leaf-selection "improvement" (used to accept/reject a candidate
// split under best-first growth, i.e. when max_leaf_nodes is set) used to read
// item.leftWeights before it was ever set for the split being evaluated --
// always 0 for a fresh node -- and separately approximated the right child's
// impurity as (imp - impLeft) instead of the real computed value. Together
// these collapse the formula to `totalWeights * impurity_left`, ignoring the
// right child (and its weight) entirely.
//
// This dataset's root has 10 points: x=1..8 map to y=1..8 (a low-variance
// group), and x=9,10 map to y=-50,50 (two points with huge spread). With
// max_leaf_nodes=2, only the root's own split is a candidate, so this is a
// pure accept/reject threshold check on a single candidate, unaffected by any
// leaf-selection-order concern (a separate bug, fixed and covered by a
// follow-up PR/commit). The best split separates {y=1..8,-50} (impurity
// ~298.02, weight 9) from {y=50} (impurity 0, weight 1). The TRUE weighted
// impurity decrease for this split is ~2392.18, while the buggy formula
// computes totalWeights(10) * impurity_left(~298.02) = ~2980.25 instead,
// wildly overestimating it since it ignores the right side entirely. Setting
// min_impurity_decrease_in_split_node to 260 (so the internal threshold,
// scaled by totalWeights=10, is ~2600) sits strictly between these two
// values: the correct formula rejects the split (single-leaf tree, every
// prediction equal to the root's mean, 3.6), while the buggy formula would
// have wrongly accepted it (verified against a build of the pre-fix code).
DF_BEST_FIRST_TEST(
    "best-first split priority uses the real impurity decrease, not a same-side approximation") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    static const float x[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    static const float y[] = { 1, 2, 3, 4, 5, 6, 7, 8, -50, 50 };

    const auto predictions =
        this->train_and_predict_all(x, y, 10, /*max_leaf_nodes*/ 2, /*min_impurity_decrease*/ 260.0);

    for (double p : predictions) {
        REQUIRE(p == Approx(3.6).epsilon(1e-6));
    }
}

} // namespace oneapi::dal::decision_forest::test
