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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/communicator.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::test::engine {

spmd::communicator<spmd::device_memory_access::none> get_global_ccl_host_communicator();
#ifdef ONEDAL_DATA_PARALLEL
spmd::communicator<spmd::device_memory_access::usm> get_global_ccl_device_communicator(
    sycl::queue&);
#endif

} // namespace oneapi::dal::test::engine
