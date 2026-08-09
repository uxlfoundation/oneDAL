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

#include "oneapi/dal/backend/primitives/stat/cov.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/loops.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include <sycl/ext/oneapi/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event means(sycl::queue& q,
                  std::int64_t row_count,
                  const ndview<Float, 1>& sums,
                  ndview<Float, 1>& means,
                  const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(means.has_mutable_data());
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, means.get_mutable_data()));
    ONEDAL_ASSERT(sums.get_dimension(0) == means.get_dimension(0));

    const auto column_count = sums.get_dimension(0);

    const Float inv_n = Float(1.0 / double(row_count));

    const Float* sums_ptr = sums.get_data();
    Float* means_ptr = means.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_1d(column_count);
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float s = sums_ptr[idx];
            means_ptr[idx] = inv_n * s;
        });
    });
}

///  A kernel that computes 2d array of covariance matrix from 2d xtx array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  q          The SYCL queue
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  sums       The input sums of size `column_count`
/// @param[in]  cov        The input xtx matrix of size `column_count` x `column_count`
/// @param[in]  bias       The input bias value
/// @param[in]  deps       Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the covariance matrix array for reading and writing
template <typename Float>
inline sycl::event compute_covariance(sycl::queue& q,
                                      std::int64_t row_count,
                                      const ndview<Float, 1>& sums,
                                      ndview<Float, 2>& cov,
                                      bool bias,
                                      const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_mutable_data()));

    const std::int64_t n = row_count;
    const std::int64_t p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));
    const Float inv_n1 = (n > 1) ? Float(1.0 / double(n - 1)) : Float(1);
    const Float multiplier = bias ? inv_n : inv_n1;

    const Float* sums_ptr = sums.get_data();
    Float* cov_ptr = cov.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<2> id) {
            const std::int64_t gi = id.get_linear_id();

            cov_ptr[gi] -= inv_n * sums_ptr[id.get_id(0)] * sums_ptr[id.get_id(1)];
            cov_ptr[gi] *= multiplier;
        });
    });
}

///  A kernel that computes 2d array of covariance matrix from 2d xtx array
///  based on the information that the input data was centering
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  q          The SYCL queue
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  sums       The input sums of size `column_count`
/// @param[in]  cov        The input xtx matrix of size `column_count` x `column_count`
/// @param[in]  bias       The input bias value
/// @param[in]  deps       Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the covariance matrix array for reading and writing
template <typename Float>
sycl::event compute_covariance_centered(sycl::queue& q,
                                        std::int64_t row_count,
                                        ndview<Float, 2>& cov,
                                        bool bias,
                                        const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, cov.get_mutable_data()));

    const std::int64_t p = cov.get_dimension(0);
    const std::int64_t n = row_count;

    const Float inv_n = Float(1.0 / double(n));
    const Float inv_n1 = (n > 1) ? Float(1.0 / double(n - 1)) : Float(1);
    const Float multiplier = bias ? inv_n : inv_n1;

    Float* cov_ptr = cov.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(make_range_1d(p * p), [=](sycl::id<1> idx) {
            cov_ptr[idx] *= multiplier;
        });
    });
}

template <typename Float>
sycl::event covariance(sycl::queue& q,
                       std::int64_t row_count,
                       const ndview<Float, 1>& sums,
                       ndview<Float, 2>& cov,
                       bool bias,
                       bool assume_centered,
                       const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_mutable_data()));

    if (assume_centered) {
        return compute_covariance_centered(q, row_count, cov, bias, deps);
    }
    else {
        return compute_covariance(q, row_count, sums, cov, bias, deps);
    }
}

template <typename Float>
sycl::event variances(sycl::queue& q,
                      const ndview<Float, 2>& cov,
                      ndview<Float, 1>& vars,
                      const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(vars.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, cov.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, vars.get_mutable_data()));

    const auto p = cov.get_dimension(0);
    const Float* cov_ptr = cov.get_data();
    Float* vars_ptr = vars.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            vars_ptr[idx] = cov_ptr[idx * p + idx];
        });
    });
}
template <typename Float>
inline sycl::event prepare_correlation(sycl::queue& q,
                                       std::int64_t row_count,
                                       const ndview<Float, 1>& sums,
                                       const ndview<Float, 2>& corr,
                                       ndview<Float, 1>& tmp,
                                       const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(corr.has_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, corr.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    const auto n = row_count;
    const auto p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));

    const Float* sums_ptr = sums.get_data();
    const Float* corr_ptr = corr.get_data();

    Float* tmp_ptr = tmp.get_mutable_data();

    const Float eps = std::numeric_limits<Float>::epsilon();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            const Float s = sums_ptr[idx];
            const Float m = inv_n * s * s;
            const Float c = corr_ptr[idx * p + idx];
            const Float v = c - m;

            // If $Var[x_i] > 0$ is close to zero, add $\varepsilon$
            // to avoid NaN/Inf in the resulting correlation matrix
            tmp_ptr[idx] = v + eps * Float(v < eps);
        });
    });
}

template <typename Float>
inline sycl::event finalize_correlation(sycl::queue& q,
                                        std::int64_t row_count,
                                        const ndview<Float, 1>& sums,
                                        const ndview<Float, 1>& tmp,
                                        ndview<Float, 2>& corr,
                                        const event_vector& deps) {
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    const auto n = row_count;
    const auto p = sums.get_count();
    const Float inv_n = Float(1.0 / double(n));

    const Float* sums_ptr = sums.get_data();
    const Float* tmp_ptr = tmp.get_mutable_data();
    Float* corr_ptr = corr.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            const std::int64_t i = idx[0];
            const std::int64_t j = idx[1];
            const std::int64_t gi = i * p + j;

            const Float is_diag = Float(i == j);

            Float c = corr_ptr[gi];
            c -= inv_n * sums_ptr[i] * sums_ptr[j];
            c *= sycl::rsqrt(tmp_ptr[i] * tmp_ptr[j]);
            corr_ptr[gi] = c * (Float(1.0) - is_diag) + is_diag;
        });
    });
}

template <typename Float>
sycl::event correlation(sycl::queue& q,
                        std::int64_t row_count,
                        const ndview<Float, 1>& sums,
                        ndview<Float, 2>& corr,
                        const event_vector& deps) {
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, sums.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));

    auto tmp = ndarray<Float, 1>::empty(q, { corr.get_dimension(0) }, sycl::usm::alloc::device);
    auto prepare_event = prepare_correlation(q, row_count, sums, corr, tmp, deps);
    auto finalize_event = finalize_correlation(q, row_count, sums, tmp, corr, { prepare_event });
    return finalize_event;
}

template <typename Float>
inline sycl::event prepare_correlation_from_covariance(sycl::queue& q,
                                                       std::int64_t row_count,
                                                       const ndview<Float, 2>& cov,
                                                       ndview<Float, 1>& tmp,
                                                       bool bias,
                                                       const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, cov.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_mutable_data()));

    const auto p = cov.get_dimension(1);

    const Float* cov_ptr = cov.get_data();

    Float* tmp_ptr = tmp.get_mutable_data();

    const Float eps = std::numeric_limits<Float>::epsilon();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            Float c = cov_ptr[idx * p + idx];

            // If $Var[x_i] > 0$ is close to zero, add $\varepsilon$
            // to avoid NaN/Inf in the resulting correlation matrix
            tmp_ptr[idx] = c + eps * Float(c < eps);
        });
    });
}

template <typename Float>
inline sycl::event finalize_correlation_from_covariance(sycl::queue& q,
                                                        std::int64_t row_count,
                                                        const ndview<Float, 2>& cov,
                                                        const ndview<Float, 1>& tmp,
                                                        ndview<Float, 2>& corr,
                                                        bool bias,
                                                        const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_data()));
    ONEDAL_ASSERT(is_known_usm(q, tmp.get_data()));

    const auto p = cov.get_dimension(1);
    const Float* tmp_ptr = tmp.get_data();
    Float* corr_ptr = corr.get_mutable_data();
    const Float* cov_ptr = cov.get_data();
    return q.submit([&](sycl::handler& cgh) {
        const auto range = make_range_2d(p, p);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            const std::int64_t i = idx[0];
            const std::int64_t j = idx[1];
            const std::int64_t gi = i * p + j;
            const Float is_diag = Float(i == j);
            Float c = cov_ptr[gi] * sycl::rsqrt(tmp_ptr[i] * tmp_ptr[j]);
            corr_ptr[gi] = c * (Float(1.0) - is_diag) + is_diag;
        });
    });
}

template <typename Float>
sycl::event correlation_from_covariance(sycl::queue& q,
                                        std::int64_t row_count,
                                        const ndview<Float, 2>& cov,
                                        ndview<Float, 2>& corr,
                                        bool bias,
                                        const event_vector& deps) {
    ONEDAL_ASSERT(cov.has_mutable_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1),
                  "Correlation matrix must be square");
    ONEDAL_ASSERT(cov.get_dimension(0) == cov.get_dimension(1), "Covariance matrix must be square");
    ONEDAL_ASSERT(is_known_usm(q, corr.get_mutable_data()));
    ONEDAL_ASSERT(is_known_usm(q, cov.get_mutable_data()));
    auto tmp = ndarray<Float, 1>::empty(q, { cov.get_dimension(0) }, sycl::usm::alloc::device);
    auto prepare_event = prepare_correlation_from_covariance(q, row_count, cov, tmp, bias, deps);
    auto finalize_event =
        finalize_correlation_from_covariance(q, row_count, cov, tmp, corr, bias, { prepare_event });
    finalize_event.wait_and_throw();
    return finalize_event;
}

/// A wrapper that computes 1d array of sums of the columns from 2d data array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  data  The input data of size `row_count` x `column_count`
/// @param[in]  assume_centered
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of sums
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the sums array for reading and writing
template <typename Float>
std::tuple<ndarray<Float, 1>, sycl::event> compute_sums(sycl::queue& queue,
                                                        const ndview<Float, 2>& data,
                                                        bool assume_centered,
                                                        const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_sums, queue);
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(data.get_dimension(1) > 0);

    const std::int64_t column_count = data.get_dimension(1);
    if (assume_centered) {
        return ndarray<Float, 1>::zeros(queue, { column_count }, sycl::usm::alloc::device);
    }
    else {
        auto sums = ndarray<Float, 1>::empty(queue, { column_count }, sycl::usm::alloc::device);
        constexpr sum<Float> binary{};
        constexpr identity<Float> unary{};
        auto sums_event = reduce_by_columns(queue, data, sums, binary, unary, deps);
        return std::make_tuple(sums, sums_event);
    }
}

/// A wrapper that computes 1d array of means of the columns from precomputed sums
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in]  sums  The input sums of size `column_count`
/// @param[in]  row_count  The number of `row_count` of the input data
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of means
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the means array for reading and writing
template <typename Float>
std::tuple<ndarray<Float, 1>, sycl::event> compute_means(sycl::queue& queue,
                                                         const ndview<Float, 1>& sums,
                                                         std::int64_t row_count,
                                                         const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_means, queue);
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(sums.get_dimension(0) > 0);

    const std::int64_t column_count = sums.get_dimension(0);
    auto data_means = ndarray<Float, 1>::empty(queue, { column_count }, sycl::usm::alloc::device);
    auto means_event = means(queue, row_count, sums, data_means, deps);
    return std::make_tuple(data_means, means_event);
}

/// A wrapper that computes the mean centered data from the input data
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The SYCL queue
/// @param[in,out]  data  The input block of the data of size `row_count` x `column_count`
/// @param[in]  means  The input means of size `column_count`
/// @param[in]  deps  Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the mean centered data array for reading and writing
template <typename Float>
sycl::event get_centered(sycl::queue& queue,
                         ndview<Float, 2>& data,
                         const ndview<Float, 1>& means,
                         const event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_centered_data, queue);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto centered_data_ptr = data.get_mutable_data();
    auto means_ptr = means.get_data();

    auto centered_event = queue.submit([&](sycl::handler& h) {
        const auto range = make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::size_t i = id[0];
            const std::size_t j = id[1];
            centered_data_ptr[i * column_count + j] -= means_ptr[j];
        });
    });
    return centered_event;
}

#define INSTANTIATE_MEANS(F)                           \
    template sycl::event means<F>(sycl::queue&,        \
                                  std::int64_t,        \
                                  const ndview<F, 1>&, \
                                  ndview<F, 1>&,       \
                                  const event_vector&);

INSTANTIATE_MEANS(float)
INSTANTIATE_MEANS(double)

#define INSTANTIATE_COV(F)                                  \
    template sycl::event covariance<F>(sycl::queue&,        \
                                       std::int64_t,        \
                                       const ndview<F, 1>&, \
                                       ndview<F, 2>&,       \
                                       bool,                \
                                       bool,                \
                                       const event_vector&);

INSTANTIATE_COV(float)
INSTANTIATE_COV(double)

#define INSTANTIATE_COR_FROM_COV(F)                                          \
    template sycl::event correlation_from_covariance<F>(sycl::queue&,        \
                                                        std::int64_t,        \
                                                        const ndview<F, 2>&, \
                                                        ndview<F, 2>&,       \
                                                        bool,                \
                                                        const event_vector&);

INSTANTIATE_COR_FROM_COV(float)
INSTANTIATE_COR_FROM_COV(double)

#define INSTANTIATE_COR(F)                                   \
    template sycl::event correlation<F>(sycl::queue&,        \
                                        std::int64_t,        \
                                        const ndview<F, 1>&, \
                                        ndview<F, 2>&,       \
                                        const event_vector&);

INSTANTIATE_COR(float)
INSTANTIATE_COR(double)

#define INSTANTIATE_VARS(F)                                \
    template sycl::event variances<F>(sycl::queue&,        \
                                      const ndview<F, 2>&, \
                                      ndview<F, 1>&,       \
                                      const event_vector&);

INSTANTIATE_VARS(float)
INSTANTIATE_VARS(double)

#define INSTANTIATE_COMPUTE_SUMS(F)                                                   \
    template std::tuple<ndarray<F, 1>, sycl::event> compute_sums(sycl::queue&,        \
                                                                 const ndview<F, 2>&, \
                                                                 bool,                \
                                                                 const event_vector&);

INSTANTIATE_COMPUTE_SUMS(float)
INSTANTIATE_COMPUTE_SUMS(double)

#define INSTANTIATE_COMPUTE_MEANS(F)                                                   \
    template std::tuple<ndarray<F, 1>, sycl::event> compute_means(sycl::queue&,        \
                                                                  const ndview<F, 1>&, \
                                                                  std::int64_t,        \
                                                                  const event_vector&);

INSTANTIATE_COMPUTE_MEANS(float)
INSTANTIATE_COMPUTE_MEANS(double)

#define INSTANTIATE_GET_CENTERED(F)                        \
    template sycl::event get_centered(sycl::queue&,        \
                                      ndview<F, 2>&,       \
                                      const ndview<F, 1>&, \
                                      const event_vector&);

INSTANTIATE_GET_CENTERED(float)
INSTANTIATE_GET_CENTERED(double)

#define INSTANTIATE_COMPUTE_COVARIANCE_CENTERED(F)                     \
    template sycl::event compute_covariance_centered<F>(sycl::queue&,  \
                                                        std::int64_t,  \
                                                        ndview<F, 2>&, \
                                                        bool,          \
                                                        const event_vector&);

INSTANTIATE_COMPUTE_COVARIANCE_CENTERED(float)
INSTANTIATE_COMPUTE_COVARIANCE_CENTERED(double)

} // namespace oneapi::dal::backend::primitives
