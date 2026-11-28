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

#include "oneapi/dal/spmd/ccl/communicator.hpp"

#include "oneapi/dal/algo/linear_regression/test/fixture.hpp"

namespace oneapi::dal::linear_regression::test {

namespace te = dal::test::engine;

/// Trains on the rows of the current rank through the oneCCL communicator;
/// every rank generates the same dataset and checks the model it gets back.
template <typename TestType>
class lr_ccl_spmd_test : public lr_test<TestType, lr_ccl_spmd_test<TestType>> {
public:
    using base_t = lr_test<TestType, lr_ccl_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using train_input_t = typename base_t::train_input_t;

    /// Two or more ranks each hold fewer rows than features, so the reference
    /// coefficients are recovered only when the ranks are reduced together.
    void generate_dimensions() {
        this->t_count_ = 307;
        this->s_count_ = 24;
        this->f_count_ = 17;
        this->r_count_ = GENERATE(1, 3);
        this->intercept_ = GENERATE(0, 1);
    }

    auto get_comm() {
        using backend_t = dal::preview::spmd::backend::ccl;
#ifdef ONEDAL_DATA_PARALLEL
        return dal::preview::spmd::make_communicator<backend_t>(this->get_queue());
#else
        return dal::preview::spmd::make_communicator<backend_t>();
#endif
    }

    template <typename Descriptor, typename... Args>
    auto train_override(const Descriptor& desc, Args&&... args) {
        auto comm = this->get_comm();
        const std::int64_t rank = comm.get_rank();
        const std::int64_t rank_count = comm.get_rank_count();

        const train_input_t input{ std::forward<Args>(args)... };
        const auto x =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), rank_count);
        const auto y =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_responses(), rank_count);

        return dal::preview::train(comm, desc, x.at(rank), y.at(rank));
    }
};

TEMPLATE_LIST_TEST_M(lr_ccl_spmd_test, "LR common flow", "[lr][spmd][ccl]", lr_types) {
#ifdef ONEDAL_DATA_PARALLEL
    // SPMD training with a queue is implemented for GPU devices only.
    SKIP_IF(this->get_policy().is_cpu());
#endif
    SKIP_IF(this->not_float64_friendly());

    this->generate(777);

    this->run_and_check_linear();
}

TEMPLATE_LIST_TEST_M(lr_ccl_spmd_test, "RR common flow", "[rr][spmd][ccl]", lr_types) {
#ifdef ONEDAL_DATA_PARALLEL
    // SPMD training with a queue is implemented for GPU devices only.
    SKIP_IF(this->get_policy().is_cpu());
#endif
    SKIP_IF(this->not_float64_friendly());

    this->generate(777);

    this->run_and_check_ridge();
}

} // namespace oneapi::dal::linear_regression::test
