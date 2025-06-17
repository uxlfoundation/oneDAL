/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/pca/infer.hpp"
#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;

template <typename TestType>
class pca_badarg_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    static constexpr std::int64_t row_count = 8;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t element_count = row_count * column_count;

    auto get_descriptor() const {
        return pca::descriptor<float, Method, pca::task::dim_reduction>{};
    }

    table get_train_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(train_data_.data(), override_row_count, override_column_count);
    }

    table get_infer_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(infer_data_.data(), override_row_count, override_column_count);
    }

private:
    static constexpr std::array<float, element_count> train_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };

    static constexpr std::array<float, element_count> infer_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };
};

using pca_types = COMBINE_TYPES((float, double), (pca::method::cov, pca::method::svd));

#define PCA_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(pca_badarg_test, name, "[pca][badarg]", pca_types)

PCA_BADARG_TEST("accepts non-negative component_count") {
    REQUIRE_NOTHROW(this->get_descriptor().set_component_count(0));
}

PCA_BADARG_TEST("throws if component_count is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_component_count(-1), domain_error);
}

PCA_BADARG_TEST("throws if train data is empty") {
    const auto pca_desc = this->get_descriptor().set_component_count(2);

    REQUIRE_THROWS_AS(this->train(pca_desc, homogen_table{}), domain_error);
}

PCA_BADARG_TEST("throws if train data columns less than component count") {
    const auto pca_desc = this->get_descriptor().set_component_count(4);

    REQUIRE_THROWS_AS(this->train(pca_desc, this->get_train_data()), invalid_argument);
}

PCA_BADARG_TEST("throws if infer data is empty") {
    const auto pca_desc = this->get_descriptor().set_component_count(2);
    const auto model = this->train(pca_desc, this->get_train_data()).get_model();

    REQUIRE_THROWS_AS(this->infer(pca_desc, model, homogen_table{}), domain_error);
}

PCA_BADARG_TEST("throws if component count neq eigenvector_rows") {
    auto pca_desc = this->get_descriptor().set_component_count(2);
    const auto model = this->train(pca_desc, this->get_train_data()).get_model();
    pca_desc.set_component_count(4);

    REQUIRE_THROWS_AS(this->infer(pca_desc, model, this->get_infer_data()), invalid_argument);
}

PCA_BADARG_TEST("throws if infer data column count neq eigenvector columns") {
    const auto pca_desc = this->get_descriptor().set_component_count(2);
    const auto model = this->train(pca_desc, this->get_train_data()).get_model();
    const auto infer_data = this->get_infer_data(4, 4);

    REQUIRE_THROWS_AS(this->infer(pca_desc, model, infer_data), invalid_argument);
}

} // namespace oneapi::dal::pca::test
