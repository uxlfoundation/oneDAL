/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel_precomputed_impl.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/misc.hpp"

#include "oneapi/dal/backend/common.hpp"
// #include "oneapi/dal/algo/pca/backend/sign_flip.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;

using task_t = task::dim_reduction;
using input_t = train_input<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
result_t train_kernel_precomputed_impl<Float>::operator()(const descriptor_t& desc,
                                                          const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();
    ONEDAL_ASSERT(data.get_column_count() > 0);
    std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());

    auto data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    sycl::event last_event;
    if (desc.get_result_options().test(result_options::vars)) {
        auto [vars, vars_event] = compute_variances(q_, data_nd);
        result.set_variances(homogen_table::wrap(vars.flatten(q_), 1, column_count));
    }
    if (desc.get_result_options().test(result_options::eigenvectors |
                                       result_options::eigenvalues)) {
        auto [eigvals, syevd_event] = syevd_computation(q_, data_nd, { last_event });

        auto [flipped_eigvals_host, flipped_eigenvectors_host] =
            flip_eigen_data_gpu(q_, eigvals, data_nd, component_count, { syevd_event });

    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(
            homogen_table::wrap(flipped_eigvals_host.flatten(q_, {}), 1, component_count));
    }

    if (desc.get_deterministic()) {
        sign_flip_gpu(q_, flipped_eigenvectors_host, {});
    }

    if (desc.get_result_options().test(result_options::eigenvectors)) {
        result.set_eigenvectors(homogen_table::wrap(flipped_eigenvectors_host.flatten(q_),
                                                    flipped_eigenvectors_host.get_dimension(0),
                                                    flipped_eigenvectors_host.get_dimension(1)));
    }
    }

    return result;
}

template class train_kernel_precomputed_impl<float>;
template class train_kernel_precomputed_impl<double>;

} // namespace oneapi::dal::pca::backend

#endif // ONEDAL_DATA_PARALLEL
