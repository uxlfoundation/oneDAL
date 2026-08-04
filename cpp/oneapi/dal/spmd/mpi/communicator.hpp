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

#include "oneapi/dal/detail/mpi/communicator.hpp"

namespace oneapi::dal::preview::spmd {

namespace backend {
struct mpi {};
} // namespace backend

template <>
inline communicator<device_memory_access::none> make_communicator<backend::mpi>() {
    return dal::detail::mpi_communicator<device_memory_access::none>{};
}

template <>
inline communicator<device_memory_access::none> make_communicator<backend::mpi>(std::int64_t comm) {
    // make_communicator takes in a int64 representation of the MPI_comm raw pointer in order
    // to be as general as possible (for all possible MPI comm sources: from other languages
    // MPICH vs Intel MPI, generalized architecture support, etc.) by using MPI_Comm_f2c.
    MPI_Comm c = MPI_Comm_f2c(static_cast<MPI_Fint>(comm));
    return dal::detail::mpi_communicator<device_memory_access::none>{ c };
}

#ifdef ONEDAL_DATA_PARALLEL
template <>
inline communicator<device_memory_access::usm> make_communicator<backend::mpi>(sycl::queue& queue) {
    return dal::detail::mpi_communicator<device_memory_access::usm>{ queue };
}

template <>
inline communicator<device_memory_access::usm> make_communicator<backend::mpi>(sycl::queue& queue,
                                                                               std::int64_t comm) {
    MPI_Comm c = MPI_Comm_f2c(static_cast<MPI_Fint>(comm));
    return dal::detail::mpi_communicator<device_memory_access::usm>{ queue, c };
}
#endif

} // namespace oneapi::dal::preview::spmd
