/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <cmath>
#include <random>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/detail/array_utils.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using convert_types =
    std::tuple<std::tuple<float, std::tuple<float, std::uint32_t, std::int64_t>>,
               std::tuple<std::int32_t, std::tuple<std::int8_t, float, std::uint64_t>>,
               std::tuple<float, std::tuple<std::int16_t, float, float, std::uint8_t>>,
               std::tuple<std::int8_t, std::tuple<std::int8_t, float, std::int8_t, float>>>;

template <typename... Types>
constexpr auto make_types_array(const std::tuple<Types...>*) {
    constexpr std::size_t count = sizeof...(Types);
    return std::array<data_type, count>{ detail::make_data_type<Types>()... };
}

template <typename... Types>
constexpr auto get_col_size(const std::tuple<Types...>*) {
    return (sizeof(Types) + ...);
}

template <typename... Types>
constexpr auto get_col_aligns(const std::tuple<Types...>*) {
    return (alignof(Types) + ...);
}

template <typename Param>
class copy_convert_fixture : public te::policy_fixture {
public:
    using result_t = std::tuple_element_t<0, Param>;
    using sources_t = std::tuple_element_t<1, Param>;

    constexpr static inline const sources_t* dummy_ptr = nullptr;
    constexpr static inline auto data_types = make_types_array(dummy_ptr);
    constexpr static inline std::int64_t col_size = get_col_size(dummy_ptr);
    constexpr static inline std::int64_t col_aligns = get_col_aligns(dummy_ptr);
    constexpr static inline std::int64_t row_count = std::tuple_size_v<sources_t>;
    constexpr static inline auto result_type = detail::make_data_type<result_t>();

    bool is_initialized() const {
        return col_count > 0l;
    }

    void check_if_initialized() const {
        if (!is_initialized()) {
            throw std::runtime_error{ "convert test is not initialized" };
        }
    }

    inline auto get_host_policy() const {
        return detail::host_policy::get_default();
    }

#ifdef ONEDAL_DATA_PARALLEL
    inline auto get_device_policy() {
        auto& queue = this->get_queue();
        return detail::data_parallel_policy{ queue };
    }
#endif // ONEDAL_DATA_PARALLEL

    void generate_input(std::int64_t seed = 777) {
        std::mt19937_64 generator(seed);
        std::uniform_int_distribution<int> dist(0, 127);

        const auto gtr_size = row_count * col_count;
        const auto inp_size = col_size * col_count + col_aligns;

        inp = dal::array<dal::byte_t>::empty(inp_size);
        gtr = dal::array<result_t>::empty(gtr_size);

        std::int64_t inp_offset = 0l, gtr_offset = 0l;
        for (std::int64_t row = 0l; row < row_count; ++row) {
            const data_type dtype = data_types.at(row);
            const auto type_size = detail::get_data_type_size(dtype);
            const auto type_align = detail::get_data_type_align(dtype);

            inp_offset = align_offset(inp_offset, type_align);
            auto* inp_raw = inp.get_mutable_data() + inp_offset;
            auto* gtr_ptr = gtr.get_mutable_data() + gtr_offset;

            backend::dispatch_by_data_type(dtype, [&](auto type) -> void {
                using type_t = std::remove_cv_t<decltype(type)>;
                auto* inp_ptr = reinterpret_cast<type_t*>(inp_raw);
                for (std::int64_t col = 0l; col < col_count; ++col) {
                    const int value = dist(generator);
                    inp_ptr[col] = static_cast<type_t>(value);
                    gtr_ptr[col] = static_cast<result_t>(inp_ptr[col]);
                }
            });

            inp_offset += (type_size * col_count);
            gtr_offset += col_count;
        }

        REQUIRE(gtr_offset == gtr_size);
#ifdef ONEDAL_DATA_PARALLEL
        auto dev_policy = get_device_policy();
        dev = detail::copy(dev_policy, inp);
#endif // ONEDAL_DATA_PARALLEL
    }

    void generate(std::int64_t seed = 777) {
        col_count = GENERATE(1, 17, 127, 1'027, 199'999);
        CAPTURE(col_count, row_count);
        generate_input(seed);
    }

    dal::array<data_type> get_types_array() const {
        auto result = dal::array<data_type>::empty(col_count);
        data_type* const res_ptr = result.get_mutable_data();
        std::copy(data_types.cbegin(), data_types.cend(), res_ptr);
        return result;
    }

    void compare_with_groundtruth_rm(const dal::array<result_t>& res) {
        REQUIRE(res.get_data() != gtr.get_data());
        const auto count = col_count * row_count;
        REQUIRE(count == gtr.get_count());
        REQUIRE(count == res.get_count());

        const result_t* const res_ptr = res.get_data();
        const result_t* const gtr_ptr = gtr.get_data();

        for (std::int64_t i = 0l; i < count; ++i) {
            const result_t res_val = res_ptr[i];
            const result_t gtr_val = gtr_ptr[i];
            if (res_val != gtr_val) {
                CAPTURE(i, res_val, gtr_val);
                REQUIRE(res_val == gtr_val);
            }
        }
    }

    void compare_with_groundtruth_cm(const dal::array<result_t>& res) {
        REQUIRE(res.get_data() != gtr.get_data());
        const auto count = col_count * row_count;
        REQUIRE(count == gtr.get_count());
        REQUIRE(count == res.get_count());

        const result_t* const res_ptr = res.get_data();
        const result_t* const gtr_ptr = gtr.get_data();

        for (std::int64_t i = 0l; i < count; ++i) {
            const std::int64_t row = i / col_count;
            const std::int64_t col = i - row * col_count;

            const std::int64_t j = row + row_count * col;

            const result_t gtr_val = gtr_ptr[i];
            const result_t res_val = res_ptr[j];

            if (res_val != gtr_val) {
                CAPTURE(i, j, row, col, res_val, gtr_val);
                REQUIRE(res_val == gtr_val);
            }
        }
    }

protected:
    std::int64_t col_count = 0l;
#ifdef ONEDAL_DATA_PARALLEL
    dal::array<dal::byte_t> dev;
#endif // ONEDAL_DATA_PARALLEL
    dal::array<dal::byte_t> inp;
    dal::array<result_t> gtr;
};

} // namespace oneapi::dal::backend::primitives::test
