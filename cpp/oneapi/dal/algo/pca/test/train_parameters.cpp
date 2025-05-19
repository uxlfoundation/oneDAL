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

#include "oneapi/dal/algo/pca/test/fixture.hpp"

#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;
namespace pca = oneapi::dal::pca;

using pca_types_cov = COMBINE_TYPES((float, double), (pca::method::cov));

template <typename TestType>
class pca_params_test : public pca_test<TestType, pca_params_test<TestType>> {
public:
    using base_t = pca_test<TestType, pca_params_test<TestType>>;

    using float_t = typename base_t::float_t;

    using input_t = typename base_t::input_t;
    using result_t = typename base_t::result_t;

    void generate_parameters() {
        this->block_ = GENERATE(64, 2048);
        this->grain_size_ = GENERATE(2, 10, 50);
        this->pack_as_struct_ = GENERATE(0, 1);
    }

    auto get_current_parameters() const {
        pca::detail::train_parameters res{};
        res.set_cpu_macro_block(this->block_);
        res.set_cpu_grain_size(this->grain_size_);
        return res;
    }

    template <typename Desc, typename... Args>
    result_t train_override(Desc&& desc, Args&&... args) {
        REQUIRE(this->block_ > 0);
        REQUIRE(this->grain_size_ > 0);
        const auto params = this->get_current_parameters();
        if (this->pack_as_struct_) {
            return te::float_algo_fixture<float_t>::train(std::forward<Desc>(desc),
                                                          params,
                                                          input_t{ std::forward<Args>(args)... });
        }
        else {
            return te::float_algo_fixture<float_t>::train(std::forward<Desc>(desc),
                                                          params,
                                                          std::forward<Args>(args)...);
        }
    }

private:
    std::int64_t block_;
    std::int64_t grain_size_;
    bool pack_as_struct_;
};

TEMPLATE_LIST_TEST_M(pca_params_test,
                     "pca dim reduction params",
                     "[pca][dim_reduction][params]",
                     pca_types_cov) {
    SKIP_IF(this->get_policy().is_gpu());

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1000, 100 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 100000, 10 }.fill_uniform(-0.2, 1.5));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto data_table_id = this->get_homogen_table_id();

    const std::int64_t component_count =
        GENERATE_COPY(1, data.get_column_count(), data.get_column_count() / 2);

    this->generate_parameters();

    this->general_checks(data, component_count, data_table_id);
}

} // namespace oneapi::dal::pca::test
