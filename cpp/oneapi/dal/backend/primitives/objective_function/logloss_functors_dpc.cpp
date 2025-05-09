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

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/objective_function/logloss_functors.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas.hpp"

namespace oneapi::dal::backend::primitives {

namespace pr = dal::backend::primitives;

std::int64_t get_block_size(std::int64_t n, std::int64_t p) {
    constexpr std::int64_t max_alloc_size = 1 << 21;
    return p > max_alloc_size ? 512 : max_alloc_size / p;
}

template <typename Float>
void logloss_hessian_product<Float>::reserve_memory() {
    raw_hessian_ = ndarray<Float, 1>::empty(q_, { n_ }, sycl::usm::alloc::device);
    buffer_ = ndarray<Float, 1>::empty(q_, { n_ }, sycl::usm::alloc::device);
    tmp_gpu_ = ndarray<Float, 1>::empty(q_, { p_ + 1 }, sycl::usm::alloc::device);
    if (data_.get_kind() == dal::csr_table::kind()) {
        sp_handle_.reset(new sparse_matrix_handle(q_));
        set_csr_data(q_, *sp_handle_, static_cast<const csr_table&>(data_));
    }
}

template <typename Float>
logloss_hessian_product<Float>::logloss_hessian_product(sycl::queue& q,
                                                        const table& data,
                                                        Float L2,
                                                        bool fit_intercept,
                                                        std::int64_t bsz)
        : q_(q),
          data_(data),
          n_(data.get_row_count()),
          p_(data.get_column_count()),
          L2_(L2),
          fit_intercept_(fit_intercept),
          bsz_(bsz == -1 ? get_block_size(n_, p_) : bsz) {
    this->reserve_memory();
}

template <typename Float>
logloss_hessian_product<Float>::logloss_hessian_product(sycl::queue& q,
                                                        comm_t comm,
                                                        const table& data,
                                                        Float L2,
                                                        bool fit_intercept,
                                                        std::int64_t bsz)
        : q_(q),
          comm_(comm),
          data_(data),
          n_(data.get_row_count()),
          p_(data.get_column_count()),
          L2_(L2),
          fit_intercept_(fit_intercept),
          bsz_(bsz == -1 ? get_block_size(n_, p_) : bsz) {
    this->reserve_memory();
}

template <typename Float>
ndview<Float, 1>& logloss_hessian_product<Float>::get_raw_hessian() {
    return raw_hessian_;
}

template <typename Float>
sycl::event logloss_hessian_product<Float>::compute_with_fit_intercept(const ndview<Float, 1>& vec,
                                                                       ndview<Float, 1>& out,
                                                                       const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_hessp_with_fit_intercept, q_);
    auto* const tmp_ptr = tmp_gpu_.get_mutable_data();
    ONEDAL_ASSERT(vec.get_dimension(0) == p_ + 1);
    ONEDAL_ASSERT(out.get_dimension(0) == p_ + 1);
    auto fill_buffer_event = fill<Float>(q_, buffer_, Float(1), deps);
    auto out_suf = out.get_slice(1, p_ + 1);
    auto tmp_suf = tmp_gpu_.slice(1, p_);
    auto out_bias = out.get_slice(0, 1);
    auto vec_suf = vec.get_slice(1, p_ + 1);
    ndview<Float, 1> tmp_ndview = tmp_gpu_;

    sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);

    const Float v0 = vec.at_device(q_, 0, deps);
    event_vector last_iter_deps = { fill_buffer_event, fill_out_event };

    if (data_.get_kind() == dal::csr_table::kind()) {
        const auto* const hess_ptr = raw_hessian_.get_data();
        auto* const out_ptr = out.get_mutable_data();
        auto* const buffer_ptr = buffer_.get_mutable_data();
        sycl::event event_xv;
        {
            event_xv = gemv(q_,
                            transpose::nontrans,
                            *sp_handle_,
                            vec_suf,
                            buffer_,
                            Float(1),
                            v0,
                            last_iter_deps);
            // to ensure sparse blas kernel stability
            event_xv.wait_and_throw();
        }

        sycl::event event_dxv = q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on({ event_xv });
            const auto range = make_range_1d(n_);
            auto sum_reduction = sycl::reduction(out_ptr, sycl::plus<>());
            cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v0) {
                buffer_ptr[idx] = buffer_ptr[idx] * hess_ptr[idx];
                sum_v0 += buffer_ptr[idx];
            });
        });
        sycl::event event_xtdxv;
        {
            event_xtdxv = gemv(q_,
                               transpose::trans,
                               *sp_handle_,
                               buffer_,
                               out_suf,
                               Float(1),
                               Float(0),
                               { event_dxv });
            // To ensure sparse blas kernel stability
            event_xtdxv.wait_and_throw();
        }
        last_iter_deps = { event_xtdxv };
    }
    else {
        const uniform_blocking blocking(n_, bsz_);
        row_accessor<const Float> data_accessor(data_);

        for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
            const auto last = blocking.get_block_end_index(b);
            const auto first = blocking.get_block_start_index(b);
            const auto length = last - first;
            auto x_rows = data_accessor.pull(q_, { first, last }, sycl::usm::alloc::device);
            auto x_nd = pr::ndarray<Float, 2>::wrap(x_rows, { length, p_ });
            auto buffer_batch = buffer_.slice(first, length);
            sycl::event event_xv =
                gemv(q_, x_nd, vec_suf, buffer_batch, Float(1), v0, last_iter_deps);
            event_xv.wait_and_throw(); // Without this line gemv does not work correctly

            auto* const buffer_ptr = buffer_batch.get_mutable_data();
            const auto* const hess_ptr = raw_hessian_.get_data() + first;

            auto fill_tmp_event = fill<Float>(q_, tmp_gpu_, Float(0), last_iter_deps);

            sycl::event event_dxv = q_.submit([&](sycl::handler& cgh) {
                cgh.depends_on({ event_xv, fill_tmp_event });
                const auto range = make_range_1d(length);
                auto sum_reduction = sycl::reduction(tmp_ptr, sycl::plus<>());
                cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v0) {
                    buffer_ptr[idx] = buffer_ptr[idx] * hess_ptr[idx];
                    sum_v0 += buffer_ptr[idx];
                });
            });

            sycl::event event_xtdxv =
                gemv(q_, x_nd.t(), buffer_batch, tmp_suf, Float(1), Float(0), { event_dxv });
            event_xtdxv.wait_and_throw(); // Without this line gemv does not work correctly

            sycl::event update_result_e =
                element_wise(q_, sycl::plus<>(), out, tmp_ndview, out, { event_xtdxv });

            last_iter_deps = { update_result_e };
        }
    }

    if (comm_.get_rank_count() > 1) {
        sycl::event::wait_and_throw(last_iter_deps);
        {
            ONEDAL_PROFILER_TASK(hessp_allreduce);
            auto hessp_arr = dal::array<Float>::wrap(q_, out.get_mutable_data(), out.get_count());
            comm_.allreduce(hessp_arr).wait();
        }
    }

    const Float regularization_factor = L2_;

    const auto kernel_regularization = [=](const Float a, const Float param) {
        return a + param * regularization_factor;
    };

    auto add_regularization_event =
        element_wise(q_, kernel_regularization, out_suf, vec_suf, out_suf, last_iter_deps);
    return add_regularization_event;
}

template <typename Float>
sycl::event logloss_hessian_product<Float>::compute_without_fit_intercept(
    const ndview<Float, 1>& vec,
    ndview<Float, 1>& out,
    const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_hessp_without_fit_intercept, q_);
    ONEDAL_ASSERT(vec.get_dimension(0) == p_);
    ONEDAL_ASSERT(out.get_dimension(0) == p_);

    ndview<Float, 1> buffer_view_ = buffer_;
    ndview<Float, 1> hess_view_ = raw_hessian_;

    sycl::event fill_out_event = fill<Float>(q_, out, Float(0), deps);

    event_vector last_iter_deps = { fill_out_event };

    if (data_.get_kind() == dal::csr_table::kind()) {
        sycl::event event_xv = gemv(q_,
                                    transpose::nontrans,
                                    *sp_handle_,
                                    vec,
                                    buffer_,
                                    Float(1),
                                    Float(0),
                                    last_iter_deps);
        event_xv.wait_and_throw(); // Without this line gemv does not work correctly

        constexpr sycl::multiplies<Float> kernel_mul{};
        auto event_dxv =
            element_wise(q_, kernel_mul, buffer_view_, hess_view_, buffer_view_, { event_xv });

        sycl::event event_xtdxv = gemv(q_,
                                       transpose::trans,
                                       *sp_handle_,
                                       buffer_,
                                       out,
                                       Float(1),
                                       Float(0),
                                       { event_dxv });
        event_xtdxv.wait_and_throw(); // Without this line gemv does not work correctly

        last_iter_deps = { event_xtdxv };
    }
    else {
        const uniform_blocking blocking(n_, bsz_);
        ndview<Float, 1> tmp_ndview = tmp_gpu_.slice(0, p_);
        row_accessor<const Float> data_accessor(data_);

        for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
            const auto last = blocking.get_block_end_index(b);
            const auto first = blocking.get_block_start_index(b);
            const auto length = last - first;
            ONEDAL_ASSERT(0l < length);
            auto x_rows = data_accessor.pull(q_, { first, last }, sycl::usm::alloc::device);
            auto x_nd = pr::ndarray<Float, 2>::wrap(x_rows, { length, p_ });
            ndview<Float, 1> buffer_batch = buffer_.slice(first, length);
            ndview<Float, 1> hess_batch = raw_hessian_.slice(first, length);

            sycl::event event_xv =
                gemv(q_, x_nd, vec, buffer_batch, Float(1), Float(0), last_iter_deps);
            event_xv.wait_and_throw(); // Without this line gemv does not work correctly

            constexpr sycl::multiplies<Float> kernel_mul{};
            auto event_dxv =
                element_wise(q_, kernel_mul, buffer_batch, hess_batch, buffer_batch, { event_xv });

            auto fill_tmp_event = fill<Float>(q_, tmp_ndview, Float(0), last_iter_deps);

            sycl::event event_xtdxv = gemv(q_,
                                           x_nd.t(),
                                           buffer_batch,
                                           tmp_ndview,
                                           Float(1),
                                           Float(0),
                                           { event_dxv, fill_tmp_event });
            event_xtdxv.wait_and_throw(); // Without this line gemv does not work correctly

            sycl::event update_grad_e =
                element_wise(q_, sycl::plus<>(), out, tmp_ndview, out, { event_xtdxv });
            last_iter_deps = { update_grad_e };
        }
    }

    if (comm_.get_rank_count() > 1) {
        {
            ONEDAL_PROFILER_TASK(hessp_allreduce);
            auto hessp_arr = dal::array<Float>::wrap(q_,
                                                     out.get_mutable_data(),
                                                     out.get_count(),
                                                     last_iter_deps);
            comm_.allreduce(hessp_arr).wait();
        }
    }

    const Float regularization_factor = L2_;

    const auto kernel_regularization = [=](const Float a, const Float param) {
        return a + param * regularization_factor;
    };

    auto add_regularization_event =
        element_wise(q_, kernel_regularization, out, vec, out, last_iter_deps);

    return add_regularization_event;
}

template <typename Float>
sycl::event logloss_hessian_product<Float>::operator()(const ndview<Float, 1>& vec,
                                                       ndview<Float, 1>& out,
                                                       const event_vector& deps) {
    if (fit_intercept_) {
        return compute_with_fit_intercept(vec, out, deps);
    }
    else {
        return compute_without_fit_intercept(vec, out, deps);
    }
}

template <typename Float>
void logloss_function<Float>::reserve_memory() {
    probabilities_ = ndarray<Float, 1>::empty(q_, { n_ }, sycl::usm::alloc::device);
    gradient_ = ndarray<Float, 1>::empty(q_, { dimension_ }, sycl::usm::alloc::device);
    buffer_ = ndarray<Float, 1>::empty(q_, { p_ + 2 }, sycl::usm::alloc::device);
    if (data_.get_kind() == dal::csr_table::kind()) {
        sp_handle_.reset(new sparse_matrix_handle(q_));
        set_csr_data(q_, *sp_handle_, static_cast<const csr_table&>(data_));
    }
}

template <typename Float>
logloss_function<Float>::logloss_function(sycl::queue& q,
                                          const table& data,
                                          const ndview<std::int32_t, 1>& labels,
                                          Float L2,
                                          bool fit_intercept,
                                          std::int64_t bsz)
        : q_(q),
          data_(data),
          labels_(labels),
          n_(data.get_row_count()),
          p_(data.get_column_count()),
          L2_(L2),
          fit_intercept_(fit_intercept),
          bsz_(bsz == -1l ? get_block_size(n_, p_) : bsz),
          dimension_(fit_intercept ? p_ + 1 : p_),
          hessp_(q, data, L2, fit_intercept, bsz_) {
    ONEDAL_ASSERT(labels.get_dimension(0) == n_);
    this->reserve_memory();
}

template <typename Float>
logloss_function<Float>::logloss_function(sycl::queue& q,
                                          comm_t comm,
                                          const table& data,
                                          const ndview<std::int32_t, 1>& labels,
                                          Float L2,
                                          bool fit_intercept,
                                          std::int64_t bsz)
        : q_(q),
          comm_(comm),
          data_(data),
          labels_(labels),
          n_(data.get_row_count()),
          p_(data.get_column_count()),
          L2_(L2),
          fit_intercept_(fit_intercept),
          bsz_(bsz == -1 ? get_block_size(n_, p_) : bsz),
          dimension_(fit_intercept ? p_ + 1 : p_),
          hessp_(q, comm, data, L2, fit_intercept, bsz_) {
    ONEDAL_ASSERT(labels.get_dimension(0) == n_);
    this->reserve_memory();
}

template <typename Float>
event_vector logloss_function<Float>::update_x(const ndview<Float, 1>& x,
                                               bool need_grad,
                                               bool need_hessp,
                                               const event_vector& deps) {
    ONEDAL_PROFILER_TASK(logloss_function_update_weights, q_);
    using dal::backend::operator+;
    value_ = 0;
    auto fill_event = fill(q_, gradient_, Float(0), deps);
    ndview<Float, 1> grad_ndview = gradient_;
    ndview<Float, 1> raw_hessian = hessp_.get_raw_hessian();
    ndview<Float, 1> loss_batch = buffer_.slice(0, 1);
    event_vector last_iter_e = { fill_event };
    constexpr Float zero(0);

    if (data_.get_kind() == dal::csr_table::kind()) {
        auto prob_e = compute_probabilities_sparse(q_,
                                                   x,
                                                   *sp_handle_,
                                                   probabilities_,
                                                   fit_intercept_,
                                                   { fill_event });

        auto fill_loss_e = fill(q_, loss_batch, zero, deps);

        sycl::event compute_e;
        if (need_grad) {
            compute_e = compute_logloss_with_der_sparse(q_,
                                                        *sp_handle_,
                                                        labels_,
                                                        probabilities_,
                                                        loss_batch,
                                                        grad_ndview,
                                                        fit_intercept_,
                                                        { fill_loss_e, prob_e });
        }
        else {
            compute_e = compute_logloss(q_,
                                        labels_,
                                        probabilities_,
                                        loss_batch,
                                        fit_intercept_,
                                        { fill_loss_e, prob_e });
        }

        value_ = loss_batch.at_device(q_, 0, { compute_e });

        last_iter_e = { compute_e };

        if (need_hessp) {
            auto hess_e = compute_raw_hessian(q_, probabilities_, raw_hessian, { prob_e });
            last_iter_e = last_iter_e + hess_e;
        }
    }
    else {
        const uniform_blocking blocking(n_, bsz_);
        ndview<Float, 1> grad_batch = buffer_.slice(1, dimension_);

        for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
            const auto first = blocking.get_block_start_index(b);
            const auto last = blocking.get_block_end_index(b);
            const std::int64_t cursize = last - first;
            ONEDAL_ASSERT(0l < cursize);

            const auto data_rows = row_accessor<const Float>(data_).pull(q_,
                                                                         { first, last },
                                                                         sycl::usm::alloc::device);
            const auto data_batch = ndarray<Float, 2>::wrap(data_rows, { cursize, p_ });
            const auto labels_batch = labels_.get_slice(first, first + cursize);
            auto prob_batch = probabilities_.slice(first, cursize);
            sycl::event prob_e =
                compute_probabilities(q_, x, data_batch, prob_batch, fit_intercept_, last_iter_e);

            auto fill_buffer_e = fill(q_, buffer_, zero, last_iter_e);

            sycl::event compute_e;
            if (need_grad) {
                compute_e = compute_logloss_with_der(q_,
                                                     data_batch,
                                                     labels_batch,
                                                     prob_batch,
                                                     loss_batch,
                                                     grad_batch,
                                                     fit_intercept_,
                                                     { fill_buffer_e, prob_e });

                sycl::event update_grad_e = element_wise(q_,
                                                         sycl::plus<>(),
                                                         grad_ndview,
                                                         grad_batch,
                                                         grad_ndview,
                                                         { compute_e });
                last_iter_e = { update_grad_e };
            }
            else {
                compute_e = compute_logloss(q_,
                                            labels_batch,
                                            prob_batch,
                                            loss_batch,
                                            fit_intercept_,
                                            { fill_buffer_e, prob_e });
                last_iter_e = { compute_e };
            }

            value_ += loss_batch.at_device(q_, 0, { compute_e });
            if (need_hessp) {
                auto raw_hessian_batch = raw_hessian.get_slice(first, first + cursize);
                auto hess_e = compute_raw_hessian(q_, prob_batch, raw_hessian_batch, { prob_e });
                last_iter_e = last_iter_e + hess_e;
            }

            // TODO: Delete this wait_and_throw
            // ensure that while event is running in the background data is not overwritten
            wait_or_pass(last_iter_e).wait_and_throw();
        }
    }
    if (comm_.get_rank_count() > 1) {
        if (need_grad) {
            ONEDAL_PROFILER_TASK(gradient_allreduce);
            auto gradient_arr = dal::array<Float>::wrap(q_,
                                                        gradient_.get_mutable_data(),
                                                        gradient_.get_count(),
                                                        last_iter_e);
            comm_.allreduce(gradient_arr).wait();
        }
        {
            ONEDAL_PROFILER_TASK(value_allreduce);
            comm_.allreduce(value_).wait();
        }
    }

    if (L2_ > 0) {
        auto fill_loss_e = fill(q_, loss_batch, Float(0), { last_iter_e });
        auto loss_ptr = loss_batch.get_mutable_data();
        auto grad_ptr = gradient_.get_mutable_data();
        auto w_ptr = x.get_data();
        Float regularization_factor = L2_;

        auto regularization_e = q_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(last_iter_e + fill_loss_e);
            const auto range = make_range_1d(p_);
            const std::int64_t st_id = fit_intercept_;
            auto sum_reduction = sycl::reduction(loss_ptr, sycl::plus<>());
            if (need_grad) {
                cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v0) {
                    const Float param = w_ptr[st_id + idx];
                    grad_ptr[st_id + idx] += regularization_factor * param;
                    sum_v0 += regularization_factor * param * param / 2;
                });
            }
            else {
                cgh.parallel_for(range, sum_reduction, [=](sycl::id<1> idx, auto& sum_v0) {
                    const Float param = w_ptr[st_id + idx];
                    sum_v0 += regularization_factor * param * param / 2;
                });
            }
        });

        value_ += loss_batch.at_device(q_, 0, { regularization_e });

        last_iter_e = { regularization_e };
    }

    return last_iter_e;
}

template <typename Float>
Float logloss_function<Float>::get_value() {
    return value_;
}
template <typename Float>
ndview<Float, 1>& logloss_function<Float>::get_gradient() {
    return gradient_;
}

template <typename Float>
base_matrix_operator<Float>& logloss_function<Float>::get_hessian_product() {
    return hessp_;
}

#define INSTANTIATE_FUNCTORS(F)                \
    template class logloss_hessian_product<F>; \
    template class logloss_function<F>;

INSTANTIATE_FUNCTORS(float)
INSTANTIATE_FUNCTORS(double)

} // namespace oneapi::dal::backend::primitives
