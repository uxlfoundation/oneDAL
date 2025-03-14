/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::detail {

namespace v2 {

template <typename Data>
bool integer_overflow_ops<Data>::is_safe_sum(const Data& first,
                                             const Data& second,
                                             Data& sum_result) {
    sum_result = first + second;
    volatile Data tmp = sum_result;
    tmp -= first;
    return tmp == second;
}

template <typename Data>
bool integer_overflow_ops<Data>::is_safe_mul(const Data& first,
                                             const Data& second,
                                             Data& mul_result) {
    mul_result = first * second;
    if (first != 0 && second != 0) {
        volatile Data tmp = mul_result;
        tmp /= first;
        return tmp == second;
    }
    return true;
}

template <typename Data>
Data integer_overflow_ops<Data>::check_sum_overflow(const Data& first, const Data& second) {
    Data op_result;
    if (!is_safe_sum(first, second, op_result)) {
        throw range_error(dal::detail::error_messages::overflow_found_in_sum_of_two_values());
    }
    return op_result;
}

template <typename Data>
Data integer_overflow_ops<Data>::check_mul_overflow(const Data& first, const Data& second) {
    Data op_result;
    if (!is_safe_mul(first, second, op_result)) {
        throw range_error(
            dal::detail::error_messages::overflow_found_in_multiplication_of_two_values());
    }
    return op_result;
}

template struct ONEDAL_EXPORT integer_overflow_ops<std::int8_t>;
template struct ONEDAL_EXPORT integer_overflow_ops<std::int16_t>;
template struct ONEDAL_EXPORT integer_overflow_ops<std::int32_t>;
template struct ONEDAL_EXPORT integer_overflow_ops<std::int64_t>;
template struct ONEDAL_EXPORT integer_overflow_ops<std::uint8_t>;
template struct ONEDAL_EXPORT integer_overflow_ops<std::uint16_t>;
template struct ONEDAL_EXPORT integer_overflow_ops<std::uint32_t>;
template struct ONEDAL_EXPORT integer_overflow_ops<std::uint64_t>;

#if defined(__APPLE__)
template struct ONEDAL_EXPORT integer_overflow_ops<std::size_t>;
#endif

} // namespace v2

} // namespace oneapi::dal::detail

namespace oneapi::dal::preview::detail {

#ifdef ONEDAL_DATA_PARALLEL
void ONEDAL_EXPORT check_if_pointer_matches_queue(const sycl::queue& q, const void* ptr) {
    if (ptr) {
        if (!dal::backend::is_known_usm(q, ptr)) {
            throw invalid_argument{ dal::detail::error_messages::unknown_usm_pointer_type() };
        }
    }
}
#endif

} // namespace oneapi::dal::preview::detail
