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

#include <array>

#include "oneapi/dal/algo/hdbscan/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::hdbscan::test {

namespace te = dal::test::engine;

template <typename TestType>
class hdbscan_badarg_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    static constexpr std::int64_t row_count = 5;

    auto get_descriptor() const {
        return hdbscan::descriptor<float, Method>(5, 5);
    }

    table get_data() const {
        return homogen_table::wrap(compute_data_.data(), compute_data_.size() / 2, 2);
    }

private:
    static constexpr std::array<float, row_count* 2> compute_data_ = { 1.0, 1.0, 2.0, 2.0, 1.0,
                                                                       2.0, 2.0, 1.0, 1.5, 1.5 };
};

using hdbscan_types = COMBINE_TYPES((float, double), (hdbscan::method::brute_force));

#define HDBSCAN_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(hdbscan_badarg_test, name, "[hdbscan][badarg]", hdbscan_types)

HDBSCAN_BADARG_TEST("accepts valid min_cluster_size") {
    REQUIRE_NOTHROW(this->get_descriptor().set_min_cluster_size(2));
}

HDBSCAN_BADARG_TEST("throws if min_cluster_size is less than 2") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_min_cluster_size(1), domain_error);
}

HDBSCAN_BADARG_TEST("accepts valid min_samples") {
    REQUIRE_NOTHROW(this->get_descriptor().set_min_samples(1));
}

HDBSCAN_BADARG_TEST("throws if min_samples is less than 1") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_min_samples(0), domain_error);
}

HDBSCAN_BADARG_TEST("throws if data is empty") {
    REQUIRE_THROWS_AS(this->compute(this->get_descriptor(), table{}), invalid_argument);
}

HDBSCAN_BADARG_TEST("accepts valid data") {
    REQUIRE_NOTHROW(this->compute(this->get_descriptor(), this->get_data()));
}

} // namespace oneapi::dal::hdbscan::test
