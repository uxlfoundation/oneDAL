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
#include <cstdlib>
#include <tuple>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/select_indexed_rows.hpp"

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

    template <ndorder ord>
    auto source() {
        auto& q = this->get_queue();
        auto res = ndarray<float_t, 2, ord>::empty(q, { m_, n_ });
        float_t counter = 0;
        for (std::int64_t r = 0; r < m_; ++r) {
            for (std::int64_t c = 0; c < n_; ++c) {
                res.at(r, c) = counter;
                counter += float_t(1);
            }
        }
        return res;
    }

    template <ndorder ord>
    auto destination() {
        auto& q = this->get_queue();
        return ndarray<float_t, 2, ord>::empty(q, { k_, n_ });
    }

    auto indices() {
        srand(m_ + n_ + k_);
        auto& q = this->get_queue();
        auto res = ndarray<std::int32_t, 1>::empty(q, k_);
        for (std::int64_t i = 0; i < k_; ++i) {
            res.at(i) = float_t(rand() % m_);
        }
        return res;
    }

    template <ndorder iord, ndorder oord>
    auto run() {
        auto& q = this->get_queue();

        const auto src = source<iord>();
        auto dst = destination<oord>();
        const auto ids = indices();

        select_indexed_rows(q, ids, src, dst).wait_and_throw();

        return dst;
    }

    template <ndorder iord, ndorder oord>
    auto gtr() {
        const auto src = source<iord>();
        auto dst = destination<oord>();
        const auto ids = indices();

        for (std::int64_t r = 0; r < k_; ++r) {
            const auto idx = ids.at(r);
            for (std::int64_t c = 0; c < n_; ++c) {
                dst.at(r, c) = src.at(idx, c);
            }
        }

        return dst;
    }

    template <ndorder iord, ndorder oord>
    void check() {
        const auto rs = run<iord, oord>();
        const auto gt = gtr<iord, oord>();
        for (std::int64_t r = 0; r < k_; ++r) {
            for (std::int64_t c = 0; c < n_; ++c) {
                CAPTURE(r, c, m_, n_, k_);
                REQUIRE(rs.at(r, c) == gt.at(r, c));
            }
        }
    }

    void check_full() {
        check<ndorder::c, ndorder::c>();
        check<ndorder::c, ndorder::f>();
        check<ndorder::f, ndorder::c>();
        check<ndorder::f, ndorder::f>();
    }

private:
    std::int64_t m_, n_, k_;
};

using selection_types = std::tuple<std::int32_t>;

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection indexed rows",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->check_full();
}

} // namespace oneapi::dal::backend::primitives::test
