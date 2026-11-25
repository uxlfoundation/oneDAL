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

#include "oneapi/dal/detail/ccl/communicator.hpp"
#include "oneapi/dal/detail/singleton.hpp"

namespace de = oneapi::dal::detail;

namespace oneapi::dal::preview::spmd {

namespace backend {
struct ccl {};
} // namespace backend

class ccl_info {
    friend class de::singleton<ccl_info>;

private:
    ccl_info() {
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        ccl::kvs::address_type main_addr;
        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            main_addr = kvs->get_address();
            MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs = ccl::create_kvs(main_addr);
        }
    }

public:
    ccl::shared_ptr_class<ccl::kvs> kvs;
    int rank;
    int rank_count;
};

template <>
inline communicator<device_memory_access::none> make_communicator<backend::ccl>() {
    auto& info = de::singleton<ccl_info>::get();
    // integral cast
    return dal::detail::ccl_communicator<device_memory_access::none>{ info.kvs,
                                                                      info.rank,
                                                                      info.rank_count };
}

#ifdef ONEDAL_DATA_PARALLEL
template <>
inline communicator<device_memory_access::usm> make_communicator<backend::ccl>(sycl::queue& queue) {
    auto& info = de::singleton<ccl_info>::get();
    return dal::detail::ccl_communicator<device_memory_access::usm>{
        queue,
        info.kvs,
        dal::detail::integral_cast<std::int64_t>(info.rank),
        dal::detail::integral_cast<std::int64_t>(info.rank_count)
    };
}
#endif

} // namespace oneapi::dal::preview::spmd
