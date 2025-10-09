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

#include "oneapi/dal/backend/primitives/sparse_blas.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <transpose t>
struct transpose_tag {
    static constexpr transpose value = t;
};

using transpose_nontrans = transpose_tag<transpose::nontrans>;
using transpose_trans = transpose_tag<transpose::trans>;

template <sparse_indexing indexing>
struct indexing_tag {
    static constexpr sparse_indexing value = indexing;
};

using indexing_zero_based = indexing_tag<sparse_indexing::zero_based>;
using indexing_one_based = indexing_tag<sparse_indexing::one_based>;

template <typename Param>
class sparse_blas_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr transpose trans_a = std::tuple_element_t<1, Param>::value;
    static constexpr ndorder bo = std::tuple_element_t<2, Param>::value;
    static constexpr ndorder co = std::tuple_element_t<3, Param>::value;
    static constexpr sparse_indexing indexing = std::tuple_element_t<4, Param>::value;

    sparse_blas_test() {
        m_ = 0;
        k_ = 0;
        p_ = 0;
    }

    void generate_dimensions() {
        m_ = GENERATE(4, 7, 100, 2000);
        k_ = GENERATE(4, 16, 31, 200);
        p_ = GENERATE(8, 15, 300, 1500);
        alloc_ = sycl::usm::alloc::device;
        CAPTURE(m_, k_, p_);
    }

    void generate_dimensions_gemv() {
        m_ = GENERATE(4, 7, 100, 2000);
        k_ = GENERATE(4, 16, 31, 200);
        p_ = GENERATE(1);
        alloc_ = sycl::usm::alloc::device;
        CAPTURE(m_, k_, p_);
    }

    /// Generate sparse CSR matrix A of the format:
    /// | 1  0  1  0  ... |
    /// | 0  1  0  1  ... |
    /// | 1  0  1  0  ... |
    /// | 0  1  0  1  ... |
    /// | ............... |
    auto A(sparse_matrix_handle& a) {
        check_if_initialized();
        auto& q = this->get_queue();

        /// Revert dimensions in the transposed case
        const std::int64_t local_m = trans_a == transpose::nontrans ? m_ : k_;
        const std::int64_t local_k = trans_a == transpose::nontrans ? k_ : m_;

        const std::int64_t value_count_in_odd_rows = (local_k + 1) / 2;
        const std::int64_t value_count_in_even_rows = local_k / 2;
        const std::int64_t odd_rows_count = (local_m + 1) / 2;
        const std::int64_t even_rows_count = local_m / 2;
        const std::int64_t element_count =
            value_count_in_odd_rows * odd_rows_count + value_count_in_even_rows * even_rows_count;

        data_ary_ = dal::array<float_t>::empty(q, element_count, alloc_);
        column_indices_ary_ = dal::array<std::int64_t>::empty(q, element_count, alloc_);
        row_offsets_ary_ = dal::array<std::int64_t>::empty(q, local_m + 1, alloc_);

        float_t* data = data_ary_.get_mutable_data();

        /// Initialize data values
        auto data_event = q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(element_count), [=](sycl::id<1> i) {
                data[i] = 1.0f;
            });
        });

        constexpr std::int64_t indexing_offset = (indexing == sparse_indexing::zero_based ? 0 : 1);

        auto column_indices_host_ary = dal::array<std::int64_t>::empty(element_count);
        std::int64_t* column_indices_host = column_indices_host_ary.get_mutable_data();

        /// Initialize column indices of the even rows
        for (std::int64_t i = 0; i < local_k; i += 2) {
            const std::int64_t column_index = i + indexing_offset;
            for (std::int64_t j = i / 2; j < element_count; j += local_k) {
                column_indices_host[j] = column_index;
            }
        }
        /// Initialize column indices of the odd rows
        for (std::int64_t i = 1; i < local_k; i += 2) {
            const std::int64_t column_index = i + indexing_offset;
            for (std::int64_t j = value_count_in_odd_rows + i / 2; j < element_count;
                 j += local_k) {
                column_indices_host[j] = column_index;
            }
        }

        auto row_offsets_host_ary = dal::array<std::int64_t>::empty(local_m + 1);
        std::int64_t* row_offsets_host = row_offsets_host_ary.get_mutable_data();

        /// Initialize row offsets of the even rows
        for (std::int64_t i = 0; i < local_m + 1; i += 2) {
            row_offsets_host[i] = (i / 2) * local_k + indexing_offset;
        }
        /// Initialize row offsets of the odd rows
        for (std::int64_t i = 1; i < local_m + 1; i += 2) {
            row_offsets_host[i] = value_count_in_odd_rows + (i / 2) * local_k + indexing_offset;
        }

        std::int64_t* column_indices = column_indices_ary_.get_mutable_data();
        auto column_indices_event = q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(column_indices, column_indices_host, element_count * sizeof(std::int64_t));
        });

        std::int64_t* row_offsets = row_offsets_ary_.get_mutable_data();
        auto row_offsets_event = q.submit([&](sycl::handler& cgh) {
            cgh.memcpy(row_offsets, row_offsets_host, (local_m + 1) * sizeof(std::int64_t));
        });
        table data_gpu = csr_table::wrap(q,
                                         data_ary_.get_data(),
                                         column_indices_ary_.get_data(),
                                         row_offsets_ary_.get_data(),
                                         local_m,
                                         local_k,
                                         indexing,
                                         { data_event, column_indices_event, row_offsets_event });
        return set_csr_data(q, a, static_cast<const csr_table&>(data_gpu));
    }

    auto B() {
        check_if_initialized();
        if (bo == ndorder::c) {
            return ndarray<float_t, 2, bo>::ones(this->get_queue(), { k_, p_ }, alloc_);
        }
        return ndarray<float_t, 2, bo>::ones(this->get_queue(), { p_, k_ }, alloc_);
    }

    auto C() {
        check_if_initialized();
        if (co == ndorder::c) {
            return ndarray<float_t, 2, co>::empty(this->get_queue(), { m_, p_ }, alloc_);
        }
        return ndarray<float_t, 2, co>::empty(this->get_queue(), { p_, m_ }, alloc_);
    }

    void test_gemm() {
        sparse_matrix_handle a(this->get_queue());
        auto a_e = A(a);
        auto [b, b_e] = B();
        auto c = C();

        gemm(this->get_queue(), trans_a, a, b, c, { a_e, b_e }).wait_and_throw();

        check_matmul(c);
    }

    void test_gemv() {
        sparse_matrix_handle a(this->get_queue());
        auto a_e = A(a);
        auto [x, x_e] = ndarray<float_t, 1>::ones(this->get_queue(), k_);
        auto y = ndarray<float_t, 1>::empty(this->get_queue(), m_);

        gemv<float_t>(this->get_queue(), trans_a, a, x, y, { a_e, x_e }).wait_and_throw();

        check_gemv(y);
    }

    void check_matmul(const ndarray<float_t, 2, co>& mat) {
        check_if_initialized();
        REQUIRE(mat.get_shape() == ndshape<2>{ m_, p_ });

        auto mat_host = mat.to_host(this->get_queue());
        const float_t* mat_ptr = mat_host.get_data();
        const std::int64_t result_even = (k_ + 1) / 2; // even rows result
        const std::int64_t result_odd = k_ / 2; // odd rows result

        for (std::int64_t i = 0; i < m_; i += 2) {
            for (std::int64_t j = 0; j < p_; j++) {
                if (std::int64_t(mat_ptr[i * p_ + j]) != result_even) {
                    CAPTURE(i * p_ + j, mat_ptr[i * p_ + j]);
                }
            }
        }
        for (std::int64_t i = 1; i < m_; i += 2) {
            for (std::int64_t j = 0; j < p_; j++) {
                if (std::int64_t(mat_ptr[i * p_ + j]) != result_odd) {
                    CAPTURE(i * p_ + j, mat_ptr[i * p_ + j]);
                }
            }
        }
        SUCCEED();
    }

    void check_gemv(const ndarray<float_t, 1>& res) {
        check_if_initialized();
        REQUIRE(res.get_count() == m_);

        auto res_host = res.to_host(this->get_queue());
        const float_t* res_ptr = res_host.get_data();
        const std::int64_t result_even = (k_ + 1) / 2; // even rows result
        const std::int64_t result_odd = k_ / 2; // odd rows result

        for (std::int64_t i = 0; i < m_; i += 2) {
            if (std::int64_t(res_ptr[i]) != result_even) {
                CAPTURE(i, res_ptr[i]);
            }
        }
        for (std::int64_t i = 1; i < m_; i += 2) {
            if (std::int64_t(res_ptr[i]) != result_odd) {
                CAPTURE(i, res_ptr[i]);
            }
        }
        SUCCEED();
    }

    bool is_initialized() const {
        return m_ > 0 && k_ > 0 && p_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "sparse gemm test is not initialized" };
        }
    }

private:
    std::int64_t m_;
    std::int64_t k_;
    std::int64_t p_;

    /// Sparse matrix A
    dal::array<float_t> data_ary_;
    dal::array<std::int64_t> column_indices_ary_;
    dal::array<std::int64_t> row_offsets_ary_;

    sycl::usm::alloc alloc_;
};

} // namespace oneapi::dal::backend::primitives::test
