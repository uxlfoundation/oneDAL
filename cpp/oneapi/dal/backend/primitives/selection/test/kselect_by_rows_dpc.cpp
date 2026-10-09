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
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_heap.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_impl.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename TestType>
class selection_by_rows_test : public te::policy_fixture {
public:
    using float_t = TestType;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    void test_selection(ndview<float_t, 2>& data,
                        std::int64_t row_count,
                        std::int64_t col_count,
                        std::int64_t k) {
        SECTION("Output of selected values") {
            ndarray<std::int32_t, 2> dummy_array;
            auto value_array = ndarray<float_t, 2>::empty(get_queue(), { row_count, k });
            kselect_by_rows<float_t> sel(get_queue(), data.get_shape(), k);
            sel(get_queue(), data, k, value_array).wait_and_throw();
            check_results<true, false>(data, value_array, dummy_array);
        }
        SECTION("Output of selected indices") {
            ndarray<float_t, 2> dummy_array;
            auto index_array = ndarray<std::int32_t, 2>::empty(get_queue(), { row_count, k });
            kselect_by_rows<float_t> sel(get_queue(), data.get_shape(), k);
            sel(get_queue(), data, k, index_array).wait_and_throw();
            check_results<false, true>(data, dummy_array, index_array);
        }

        SECTION("Output of both") {
            auto value_array = ndarray<float_t, 2>::empty(get_queue(), { row_count, k });
            auto index_array = ndarray<std::int32_t, 2>::empty(get_queue(), { row_count, k });
            kselect_by_rows<float_t> sel(get_queue(), data.get_shape(), k);
            sel(get_queue(), data, k, value_array, index_array).wait_and_throw();
            check_results<true, true>(data, value_array, index_array);
            auto selct_array = ndarray<float_t, 2>::empty(get_queue(), { row_count, k });
            select_indexed(get_queue(), index_array, data, selct_array).wait_and_throw();
            check_equal(value_array, selct_array);
        }
    }

    void check_equal(const ndview<float_t, 2>& res, const ndview<float_t, 2>& gtr) {
        REQUIRE(res.get_shape() == gtr.get_shape());
        const auto m = res.get_dimension(0);
        const auto k = res.get_dimension(1);
        for (std::int32_t i = 0; i < k; ++i) {
            for (std::int32_t j = 0; j < m; ++j) {
                const auto r = res.at(j, i);
                const auto g = gtr.at(j, i);
                CAPTURE(i, j, r, g);
                REQUIRE(r == g);
            }
        }
    }

    template <bool selection_out, bool indices_out>
    void check_results(const ndview<float_t, 2>& data,
                       const ndview<float_t, 2>& selection,
                       const ndview<std::int32_t, 2>& indices) {
        ONEDAL_ASSERT(!selection_out || data.get_dimension(0) == selection.get_dimension(0));
        ONEDAL_ASSERT(!indices_out || data.get_dimension(0) == indices.get_dimension(0));
        auto k = selection.get_dimension(1);
        auto row_size = data.get_dimension(1);
        auto row_count = data.get_dimension(0);
        for (std::int64_t i = 0; i < row_count; i++) {
            auto max_val = std::numeric_limits<float_t>::lowest();
            for (std::int64_t j = 0; j < k; j++) {
                float_t cur_val = get_value<selection_out, indices_out>(data,
                                                                        selection,
                                                                        indices,
                                                                        k,
                                                                        row_size,
                                                                        i,
                                                                        j);
                check_presence_in_data<selection_out, indices_out>(data,
                                                                   selection,
                                                                   indices,
                                                                   k,
                                                                   row_size,
                                                                   i,
                                                                   j,
                                                                   cur_val);
                if (max_val < cur_val)
                    max_val = cur_val;
            }
            for (std::int64_t j = 0; j < row_size; j++) {
                float_t cur_val = data.get_data()[i * row_size + j];
                if (cur_val < max_val) {
                    check_presence_in_selection<selection_out, indices_out>(selection,
                                                                            indices,
                                                                            k,
                                                                            row_size,
                                                                            i,
                                                                            j,
                                                                            cur_val);
                }
            }
        }
    }

    template <bool selection_out, bool indices_out>
    float_t get_value(const ndview<float_t, 2>& data,
                      const ndview<float_t, 2>& selection,
                      const ndview<std::int32_t, 2>& indices,
                      std::int64_t k,
                      std::int64_t row_size,
                      std::int64_t row,
                      std::int64_t pos) {
        if constexpr (selection_out) {
            return selection.get_data()[row * k + pos];
        }
        if constexpr (indices_out) {
            auto cur_index = indices.get_data()[row * k + pos];
            REQUIRE(cur_index > -1);
            REQUIRE(cur_index < row_size);
            return data.get_data()[row * row_size + cur_index];
        }
    }

    template <bool selection_out, bool indices_out>
    void check_presence_in_data(const ndview<float_t, 2>& data,
                                const ndview<float_t, 2>& selection,
                                const ndview<std::int32_t, 2>& indices,
                                std::int64_t k,
                                std::int64_t row_size,
                                std::int64_t row,
                                std::int64_t pos,
                                float_t cur_val) {
        CAPTURE(row, k, pos, cur_val);
        if constexpr (indices_out && selection_out) {
            REQUIRE(selection.get_data()[row * k + pos] ==
                    data.get_data()[row * row_size + indices.get_data()[row * k + pos]]);
        }
        std::int64_t count = 0;
        for (std::int64_t i = 0; i < row_size; i++) {
            float_t probe = data.get_data()[row * row_size + i];
            count += (std::int64_t)(probe == cur_val);
        }
        REQUIRE(count > 0);
    }

    template <bool selection_out, bool indices_out>
    void check_presence_in_selection(const ndview<float_t, 2>& selection,
                                     const ndview<std::int32_t, 2>& indices,
                                     std::int64_t k,
                                     std::int64_t row_size,
                                     std::int64_t row,
                                     std::int64_t pos,
                                     float_t cur_val) {
        CAPTURE(row, k, pos, cur_val);
        std::int64_t count = 0;
        if constexpr (!indices_out) {
            for (std::int64_t l = 0; l < k; l++) {
                count += (std::int64_t)(selection.get_data()[row * k + l] == cur_val);
            }
        }
        else {
            for (std::int64_t l = 0; l < k; l++) {
                count += (std::int64_t)(indices.get_data()[row * k + l] == pos);
            }
        }
        REQUIRE(count > 0);
    }
};

using selection_types = std::tuple<float /*, double*/>;

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection degenerated test (k == 1)",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(!this->get_policy().has_native_float64());
    using float_t = TestType;
    float_t data[] = { -2.0f, 5.0f, -3.0f, 3.0f, 4.0f, 1.0f, 1.0f, 4.0f,
                       4.0f,  1.0f, 1.0f,  0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<float_t, 2>::empty(this->get_queue(), { 3, 5 });
    auto event = this->get_queue().submit([&](sycl::handler& cgh) {
        cgh.memcpy(data_array.get_mutable_data(), data, sizeof(float_t) * 15);
    });
    event.wait();
    this->test_selection(data_array, 3, 5, 1);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection single row test (k == 2)",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(!this->get_policy().has_native_float64());
    using float_t = TestType;
    float_t data[] = { 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f,
                       0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<float_t, 2>::empty(this->get_queue(), { 1, 15 });
    auto event = this->get_queue().submit([&](sycl::handler& cgh) {
        cgh.memcpy(data_array.get_mutable_data(), data, sizeof(float_t) * 15);
    });
    event.wait();
    this->test_selection(data_array, 1, 15, 2);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection degenerated test (k == m)",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(!this->get_policy().has_native_float64());
    using float_t = TestType;
    float_t data[] = { 7.0f, 5.0f, 0.0f, 0.0f,  0.0f, 1.0f, 1.0f, 4.0f,
                       0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<float_t, 2>::empty(this->get_queue(), { 3, 5 });
    auto event = this->get_queue().submit([&](sycl::handler& cgh) {
        cgh.memcpy(data_array.get_mutable_data(), data, sizeof(float_t) * 15);
    });
    event.wait();

    this->test_selection(data_array, 3, 5, 5);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test (all zeroes)",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(!this->get_policy().has_native_float64());
    using float_t = TestType;
    auto [data_array, event] = ndarray<float_t, 2>::zeros(this->get_queue(), { 3, 5 });
    event.wait();
    this->test_selection(data_array, 3, 5, 2);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test on random data",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(!this->get_policy().has_native_float64());
    using float_t = TestType;
    std::int64_t rows = 17;
    std::int64_t cols = 33;
    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_selection(data_array, rows, cols, 31);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test on single random row (k > 32)",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(!this->get_policy().has_native_float64());
    using float_t = TestType;
    std::int64_t rows = 1;
    std::int64_t cols = 35;
    std::int64_t k = 33;
    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_selection(data_array, rows, cols, k);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test on random block (k > 32)",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(!this->get_policy().has_native_float64());
    using float_t = TestType;
    std::int64_t rows = 17;
    std::int64_t cols = 35;
    std::int64_t k = 33;
    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_selection(data_array, rows, cols, k);
}

// The input holds more than 2^31 - 1 elements, so every row past the int32 boundary is read
// through a 64-bit offset. About 26 GB of device memory with the quick-select scratch, hence
// the `[weekly]` tag.
class kselect_by_rows_overflow_test : public te::float_algo_fixture<float> {
public:
    static constexpr std::int64_t col_count = 32768;
    static constexpr std::int64_t row_count =
        std::numeric_limits<std::int32_t>::max() / col_count + 2;

    bool fits_on_device() {
        const std::int64_t needed = 3 * sizeof(float) * row_count * col_count;
        return device_global_mem_size(this->get_queue()) > needed + needed / 4;
    }

    // Each row is a permutation of 0 .. col_count - 1, so its k smallest are 0 .. k - 1.
    ndarray<float, 2> make_input() {
        auto& q = this->get_queue();
        auto data = ndarray<float, 2>::empty(q, { row_count, col_count }, sycl::usm::alloc::device);
        float* const ptr = data.get_mutable_data();
        parallel_for_2d_by_row_blocks(q,
                                      row_count,
                                      col_count,
                                      [=](std::int64_t row, std::int64_t col) {
                                          ptr[row * col_count + col] =
                                              float((col * 7919 + row) % col_count);
                                      })
            .wait_and_throw();
        return data;
    }

    void check_rows(const ndarray<float, 2>& data, std::int64_t k) {
        auto& q = this->get_queue();
        auto selection = ndarray<float, 2>::empty(q, { row_count, k }, sycl::usm::alloc::device);
        auto indices =
            ndarray<std::int32_t, 2>::empty(q, { row_count, k }, sycl::usm::alloc::device);

        kselect_by_rows<float> select(q, data.get_shape(), k);
        select(q, data, k, selection, indices).wait_and_throw();

        const auto selection_host = selection.to_host(q);
        const auto indices_host = indices.to_host(q);
        const float* const sel = selection_host.get_data();
        const std::int32_t* const ids = indices_host.get_data();

        for (const std::int64_t row : { std::int64_t(0), row_count - 2, row_count - 1 }) {
            double sum = 0;
            float max_value = 0;
            for (std::int64_t j = 0; j < k; ++j) {
                const float value = sel[row * k + j];
                const std::int64_t col = ids[row * k + j];
                REQUIRE(float((col * 7919 + row) % col_count) == value);
                sum += value;
                max_value = std::max(max_value, value);
            }
            REQUIRE(max_value == float(k - 1));
            REQUIRE(sum == double(k) * double(k - 1) / 2);
        }
    }
};

TEST_M(kselect_by_rows_overflow_test,
       "kselect_by_rows reads rows past the int32 element range",
       "[kselect][overflow][weekly]") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(!this->fits_on_device());

    auto& q = this->get_queue();
    const std::int64_t simd_k = 8;
    const std::int64_t heap_k = 64;
    const std::int64_t quick_k = get_heap_max_k<float>(q) + 1;
    REQUIRE(get_kselect_by_rows_kind<float>(q, simd_k) == kselect_by_rows_kind::simd);
    REQUIRE(get_kselect_by_rows_kind<float>(q, quick_k) == kselect_by_rows_kind::quick);

    const auto data = this->make_input();
    const std::int64_t k = GENERATE_COPY(simd_k, heap_k, quick_k);
    INFO("k = " << k);
    this->check_rows(data, k);
}

} // namespace oneapi::dal::backend::primitives::test
