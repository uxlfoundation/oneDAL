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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Compute means
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] means The [p] means for each feature
template <typename Float>
sycl::event means(sycl::queue& queue,
                  std::int64_t row_count,
                  const ndview<Float, 1>& sums,
                  ndview<Float, 1>& means,
                  const event_vector& deps = {});

/// Computes covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[in]  bias If true biased covariance estimated by maximum likelihood method computed
/// @param[out] cov  The [p x p] covariance matrix
template <typename Float>
sycl::event covariance(sycl::queue& q,
                       std::int64_t row_count,
                       const ndview<Float, 1>& sums,
                       ndview<Float, 2>& cov,
                       bool bias,
                       bool assume_centered,
                       const event_vector& deps = {});

/// A kernel that computes 2d array of covariance matrix from 2d xtx array
/// based on the information that the input data was centered
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]      q          The SYCL queue
/// @param[in]      row_count  The number of `row_count` of the input data
/// @param[in,out]  cov        The input cross-product matrix of size `column_count` x `column_count`
/// @param[in]      bias       The input bias value
/// @param[in]      deps       Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the covariance matrix array for reading and writing
template <typename Float>
sycl::event compute_covariance_centered(sycl::queue& q,
                                        std::int64_t row_count,
                                        ndview<Float, 2>& cov,
                                        bool bias,
                                        const event_vector& deps = {});

/// Compute variances
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  cov  The [p x p] covariance matrix
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] vars The [p] vars for each feature
template <typename Float>
sycl::event variances(sycl::queue& queue,
                      const ndview<Float, 2>& cov,
                      ndview<Float, 1>& vars,
                      const event_vector& deps = {});

/// Computes correlation matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] corr  The [p x p] correlation matrix
/// @param[out] tmp   The [p] temporary buffer
template <typename Float>
sycl::event correlation(sycl::queue& q,
                        std::int64_t row_count,
                        const ndview<Float, 1>& sums,
                        ndview<Float, 2>& corr,
                        const event_vector& deps = {});

/// Computes correlation matrix from covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[out] cov   The [p x p] covariance matrix
/// @param[out] corr  The [p x p] correlation matrix
/// @param[in]  bias  Determines if provided covariance estimation biased
/// @param[in]  deps  Events indicating availability of the `cov` and `corr` for reading or writing
template <typename Float>
sycl::event correlation_from_covariance(sycl::queue& q,
                                        const ndview<Float, 2>& cov,
                                        ndview<Float, 2>& corr,
                                        bool bias,
                                        const event_vector& deps = {});

/// A wrapper that computes 1d array of sums of the columns from 2d data array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue            The SYCL queue
/// @param[in]  data             The input data of size `row_count` x `column_count`
/// @param[in]  assume_centered  If true, the input data is assumed to be centered
/// @param[in]  deps             Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of sums
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the sums array for reading and writing
template <typename Float>
std::tuple<ndarray<Float, 1>, sycl::event> compute_sums(sycl::queue& queue,
                                                        const ndview<Float, 2>& data,
                                                        bool assume_centered = false,
                                                        const event_vector& deps = {});

/// A wrapper that computes 1d array of means of the columns from precomputed sums
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue     The SYCL queue
/// @param[in]  sums      The input sums of size `column_count`
/// @param[in]  row_count The number of `row_count` of the input data
/// @param[in]  deps      Events indicating availability of the `data` for reading or writing
///
/// @return A tuple of two elements, where the first element is the resulting 1d array of means
/// of size `column_count` and the second element is a SYCL event indicating the availability
/// of the means array for reading and writing
template <typename Float>
std::tuple<ndarray<Float, 1>, sycl::event> compute_means(sycl::queue& queue,
                                                         const ndview<Float, 1>& sums,
                                                         std::int64_t row_count,
                                                         const event_vector& deps = {});

/// A wrapper that computes the mean centered data from the input data
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]      queue The SYCL queue
/// @param[in,out]  data  The block of data to be centered of size `row_count` x `column_count`
/// @param[in]      means The input means of size `column_count`
/// @param[in]      deps  Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the mean centered data array for reading and writing
template <typename Float>
sycl::event get_centered(sycl::queue& queue,
                         ndview<Float, 2>& data,
                         const ndview<Float, 1>& means,
                         const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
