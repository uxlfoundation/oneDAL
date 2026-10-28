/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <type_traits>
#include <tuple>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/select_indexed.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

template <typename TestType>
class selection_by_rows_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    void generate() {
        m_ = GENERATE(1, 3, 129, 515);
        n_ = GENERATE(1, 4, 131, 517);
        k_ = GENERATE(1, 5, 133, 513, 1031);
    }

    void test_selection_1d() {
        auto [dst, dst_event] = destination();
        auto ids = indices();
        auto src = source_1d();

        select_indexed(this->get_queue(), ids, src, dst, { dst_event }).wait_and_throw();

        check_selection_1d(ids, src, dst);
    }

    void test_selection_2d() {
        auto [dst, dst_event] = destination();
        auto ids = indices();
        auto src = source_2d();

        select_indexed(this->get_queue(), ids, src, dst, { dst_event }).wait_and_throw();

        check_selection_2d(ids, src, dst);
    }

    void check_selection_1d(const ndview<std::int32_t, 2>& ids,
                            const ndview<TestType, 1>& src,
                            const ndview<TestType, 2>& dst) {
        for (std::int64_t j = 0; j < m_; ++j) {
            for (std::int64_t i = 0; i < n_; ++i) {
                const std::int64_t idx = (i + j) % k_;
                const std::int64_t val = k_ - idx;
                if (TestType(val) != dst.at(j, i)) {
                    CAPTURE(i, j, idx, val);
                    CAPTURE(ids.at(j, i), dst.at(j, i));
                    REQUIRE(TestType(val) == dst.at(j, i));
                }
            }
        }
    }

    void check_selection_2d(const ndview<std::int32_t, 2>& ids,
                            const ndview<TestType, 2>& src,
                            const ndview<TestType, 2>& dst) {
        for (std::int64_t j = 0; j < m_; ++j) {
            for (std::int64_t i = 0; i < n_; ++i) {
                const std::int64_t idx = (i + j) % k_;
                const std::int64_t val = j + idx;
                if (TestType(val) != dst.at(j, i)) {
                    CAPTURE(i, j, idx, val);
                    CAPTURE(ids.at(j, i), dst.at(j, i));
                    REQUIRE(TestType(val) == dst.at(j, i));
                }
            }
        }
    }

    auto destination() {
        return ndarray<TestType, 2>::zeros(this->get_queue(), { m_, n_ });
    }

    auto indices() {
        auto res = ndarray<std::int32_t, 2>::empty(this->get_queue(), { m_, n_ });
        for (std::int64_t j = 0; j < m_; ++j) {
            for (std::int64_t i = 0; i < n_; ++i) {
                res.at(j, i) = (i + j) % k_;
            }
        }
        return res;
    }

    auto source_1d() {
        auto res = ndarray<TestType, 1>::empty(this->get_queue(), { k_ });
        for (std::int64_t i = 0; i < k_; ++i) {
            *(res.get_mutable_data() + i) = TestType(k_ - i);
        }
        return res;
    }

    auto source_2d() {
        auto res = ndarray<TestType, 2>::empty(this->get_queue(), { m_, k_ });
        for (std::int64_t j = 0; j < m_; ++j) {
            for (std::int64_t i = 0; i < k_; ++i) {
                res.at(j, i) = TestType(i + j);
            }
        }
        return res;
    }

private:
    std::int64_t m_, n_, k_;
};

using selection_types = std::tuple<std::int32_t>;

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection indexed 1D",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_selection_1d();
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection indexed 2D",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_selection_2d();
}

} // namespace oneapi::dal::backend::primitives::test
