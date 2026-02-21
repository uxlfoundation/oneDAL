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

#pragma once

#include "oneapi/dal/algo/spectral_embedding/compute.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/detail/debug.hpp"
#include <random>

namespace oneapi::dal::spectral_embedding::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace sp_emb = oneapi::dal::spectral_embedding;

using dal::detail::operator<<;

template <typename TestType, typename Derived>
class spectral_embedding_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using input_t = sp_emb::compute_input<>;
    using result_t = sp_emb::compute_result<>;
    using descriptor_t = sp_emb::descriptor<Float, Method>;

    auto get_descriptor(std::int64_t component_count,
                        std::int64_t neighbor_count,
                        sp_emb::result_option_id compute_mode) const {
        return descriptor_t()
            .set_component_count(component_count)
            .set_neighbor_count(neighbor_count)
            .set_result_options(compute_mode);
    }

    void gen_input() {
        std::mt19937 rnd(2007 + n_ + p_ + n_ * p_);
        const te::dataframe data_df =
            GENERATE_DATAFRAME(te::dataframe_builder{ n_, p_ }.fill_normal(-0.5, 0.5, 7777));
        data_ = data_df.get_table(this->get_policy(), this->get_homogen_table_id());
    }

    void test_gold_input(Float tol = 1e-5) {
        constexpr std::int64_t n = 8;
        constexpr std::int64_t p = 4;
        constexpr std::int64_t neighbor_count = 5;
        constexpr std::int64_t component_count = 4;

        constexpr Float data[n * p] = { 0.49671415,  -0.1382643,  0.64768854,  1.52302986,
                                        -0.23415337, -0.23413696, 1.57921282,  0.76743473,
                                        -0.46947439, 0.54256004,  -0.46341769, -0.46572975,
                                        0.24196227,  -1.91328024, -1.72491783, -0.56228753,
                                        -1.01283112, 0.31424733,  -0.90802408, -1.4123037,
                                        1.46564877,  -0.2257763,  0.0675282,   -1.42474819,
                                        -0.54438272, 0.11092259,  -1.15099358, 0.37569802,
                                        -0.60063869, -0.29169375, -0.60170661, 1.85227818 };

        constexpr Float gth_embedding[n * component_count] = {
            -0.353553391, 0.442842965,     0.190005876,  0.705830111,   -0.353553391, 0.604392576,
            -0.247517958, -0.595235173,    -0.353553391, -0.391745507,  0.0443633719, -0.150208165,
            -0.353553391, -0.142548722,    0.0125222995, -0.0318482841, -0.353553391, -0.499390711,
            -0.20194266,  -0.000639679859, -0.353553391, 0.00809834849, -0.683462258, 0.273398265,
            -0.353553391, -0.0977843445,   0.449358299,  0.0195905172,  -0.353553391, 0.0761353959,
            0.436673029,  -0.220887591
        };

        constexpr Float gth_eigen_vals[n] = { 0,          3.32674524, 4.70361338, 5.26372220,
                                              5.69343808, 6.63074948, 6.80173994, 7.57999167 };

        auto desc = get_descriptor(
            component_count,
            neighbor_count,
            sp_emb::result_options::embedding | sp_emb::result_options::eigen_values);

        table data_ = homogen_table::wrap(data, n, p);

        INFO("run compute");
        auto compute_result = this->compute(desc, data_);
        auto embedding = compute_result.get_embedding();
        // std::cout << "Output" << std::endl;
        // std::cout << embedding << std::endl;

        array<Float> emb_arr = row_accessor<const Float>(embedding).pull({ 0, -1 });
        for (int j = 0; j < component_count; ++j) {
            Float diff = 0, diff_rev = 0;
            for (int i = 0; i < n; ++i) {
                Float val = emb_arr[i * component_count + j];
                Float gth_val = gth_embedding[i * component_count + j];
                diff = std::max(diff, std::abs(val - gth_val));
                diff_rev = std::max(diff_rev, std::abs(val + gth_val));
            }
            REQUIRE((diff < tol || diff_rev < tol));
        }

        auto eigen_values = compute_result.get_eigen_values();
        // std::cout << "Eigen values:" << std::endl;
        // std::cout << eigen_values << std::endl;

        array<Float> eig_val_arr = row_accessor<const Float>(eigen_values).pull({ 0, -1 });
        for (int i = 0; i < n; ++i) {
            REQUIRE(std::abs(eig_val_arr[i] - gth_eigen_vals[i]) < tol);
        }
    }

protected:
    std::int64_t n_;
    std::int64_t p_;
    table data_;
};

using spectral_embedding_types = COMBINE_TYPES((float, double), (sp_emb::method::dense_batch));

} // namespace oneapi::dal::spectral_embedding::test
