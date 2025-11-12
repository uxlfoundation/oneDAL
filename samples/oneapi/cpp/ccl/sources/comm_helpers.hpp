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

#ifdef ONEDAL_DATA_PARALLEL

#include <sycl/sycl.hpp>
#include "oneapi/dal/table/row_accessor.hpp"

template <typename Comm, typename D = float>
dal::table combine_tables(const Comm& comm, const dal::table& t) {
    sycl::queue queue = comm.get_queue();
    const std::int64_t row_count = t.get_row_count();
    const std::int64_t column_count = t.get_column_count();
    const std::int64_t count = row_count * column_count;
    std::int64_t total_count = count;
    comm.allreduce(total_count).wait();

    auto recv_buffer = dal::array<D>::empty(queue, total_count, sycl::usm::alloc::device);
    auto recv_counts = dal::array<std::int64_t>::zeros(comm.get_rank_count());
    recv_counts.get_mutable_data()[comm.get_rank()] = count;
    comm.allreduce(recv_counts).wait();

    auto displs = dal::array<std::int64_t>::empty(comm.get_rank_count());
    std::int64_t offset = 0;
    for (std::int64_t i = 0; i < comm.get_rank_count(); i++) {
        displs.get_mutable_data()[i] = offset;
        offset += recv_counts[i];
    }

    auto send_buffer =
        dal::row_accessor<const D>{ t }.pull(queue, dal::range{ 0, -1 }, sycl::usm::alloc::device);
    comm.allgatherv(send_buffer, recv_buffer, recv_counts.get_data(), displs.get_data()).wait();
    return dal::homogen_table::wrap(recv_buffer, total_count, column_count);
}

#endif
