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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_dense_misc.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float, bs_list List>
class compute_kernel_dense_impl {
    using method_t = method::dense;
    using task_t = task::compute;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using input_t = compute_input<task_t>;
    using result_t = compute_result<task_t>;
    using local_result_t = local_result<Float, List>;
    using local_buffer_list_t = local_buffer_list<Float, List>;
    using descriptor_t = detail::descriptor_base<task_t>;

public:
    compute_kernel_dense_impl(const bk::context_gpu& ctx)
            : q_(ctx.get_queue()),
              comm_(ctx.get_communicator()) {}
    result_t operator()(const descriptor_t& desc, const input_t& input);

private:
    std::int64_t get_row_block_count(std::int64_t row_count);
    std::int64_t get_column_block_count(std::int64_t column_count);
    template <bool use_weights>
    std::tuple<local_result_t, sycl::event> compute_single_pass(
        const pr::ndview<Float, 2>& data,
        sycl::usm::alloc data_alloc_kind,
        const pr::ndview<Float, 2>& weights = {});
    template <bool use_weights>
    std::tuple<local_result_t, sycl::event> compute_by_blocks(
        const pr::ndview<Float, 2>& data,
        sycl::usm::alloc data_alloc_kind,
        std::int64_t row_block_count,
        const pr::ndview<Float, 2>& weights = {});
    std::tuple<local_result_t, sycl::event> merge_blocks(local_buffer_list_t&& ndbuf,
                                                         std::int64_t column_count,
                                                         std::int64_t block_count,
                                                         sycl::usm::alloc result_alloc_kind,
                                                         const bk::event_vector& deps = {});

    std::tuple<local_result_t, sycl::event> finalize(local_result_t&& ndres,
                                                     std::int64_t row_count,
                                                     std::int64_t column_count,
                                                     const bk::event_vector& deps = {});

    result_t get_result(const descriptor_t& desc,
                        const local_result_t& ndres,
                        std::int64_t column_count,
                        alloc_kind result_alloc_kind,
                        const bk::event_vector& deps = {});

    sycl::queue q_;
    comm_t comm_;
};

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL
