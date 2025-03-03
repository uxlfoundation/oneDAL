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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/sort.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename ClassType = std::int32_t>
class uniform_voting {
public:
    virtual sycl::event operator()(const ndview<ClassType, 2>& responses,
                                   ndview<ClassType, 1>& results,
                                   const event_vector& deps = {}) = 0;
    virtual ~uniform_voting();

protected:
    uniform_voting(sycl::queue& q);
    sycl::queue& get_queue() const;

private:
    sycl::queue& queue_;
};

/// TODO: Fix small_k_voting implemenmtation

template <typename ClassType = std::int32_t>
class large_k_uniform_voting : public uniform_voting<ClassType> {
    using base_t = uniform_voting<ClassType>;

public:
    large_k_uniform_voting(sycl::queue& queue, std::int64_t max_block, std::int64_t k_response);
    sycl::event operator()(const ndview<ClassType, 2>& responses,
                           ndview<ClassType, 1>& results,
                           const event_vector& deps = {}) final;

private:
    sycl::event select_winner(ndview<ClassType, 1>& results, const event_vector& deps) const;

    ndarray<ClassType, 2> swp_, out_;
    radix_sort<ClassType> sorting_;
};

template <typename ClassType = std::int32_t>
std::unique_ptr<uniform_voting<ClassType>> make_uniform_voting(sycl::queue& queue,
                                                               std::int64_t max_block,
                                                               std::int64_t k_response);

#endif

} // namespace oneapi::dal::backend::primitives
