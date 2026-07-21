/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/table/common.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename... Types>
inline dal::array<data_type> find_array_dtypes() {
    constexpr std::size_t raw = sizeof...(Types);
    const auto count = detail::integral_cast<std::int64_t>(raw);

    auto dtypes = dal::array<data_type>::empty(count);
    auto* const dt_ptr = dtypes.get_mutable_data();

    std::int64_t feature = 0l;

    detail::apply(
        [&](const auto& type) -> void {
            using type_t = std::decay_t<decltype(type)>;
            dt_ptr[feature++] = make_data_type<type_t>();
        },
        Types{}...);

    ONEDAL_ASSERT(feature == count);

    return dtypes;
}

template <typename... Types>
inline dal::array<feature_type> find_array_ftypes() {
    constexpr std::size_t raw = sizeof...(Types);
    const auto count = detail::integral_cast<std::int64_t>(raw);

    auto ftypes = dal::array<feature_type>::empty(count);
    auto* const ft_ptr = ftypes.get_mutable_data();

    const auto make_feature_type = [](const auto& type) {
        return std::is_integral_v<std::decay_t<decltype(type)>> ? feature_type::ordinal
                                                                : feature_type::ratio;
    };

    std::int64_t feature = 0l;

    detail::apply(
        [&](const auto& type) -> void {
            using type_t = std::decay_t<decltype(type)>;
            ft_ptr[feature++] = make_feature_type(type_t{});
        },
        Types{}...);

    ONEDAL_ASSERT(feature == count);

    return ftypes;
}

template <typename... Types>
inline table_metadata make_default_metadata(const alloc_kind kind = alloc_kind::non_usm) {
    const auto dtypes = find_array_dtypes<Types...>();
    const auto ftypes = find_array_ftypes<Types...>();
    return table_metadata(dtypes, ftypes, kind);
}

/// Deduces a single allocation kind shared by all the input arrays.
/// The allocation kind of the first array is taken as the reference and all
/// the remaining arrays are required to have the same allocation kind.
/// @pre All arrays are allocated with the same allocation kind.
template <typename... Arrays>
inline alloc_kind deduce_common_alloc_kind(const Arrays&... arrays) {
    if constexpr (sizeof...(Arrays) == 0ul) {
        return alloc_kind::non_usm;
    }
    else {
        bool first = true;
        alloc_kind common = alloc_kind::non_usm;

        detail::apply(
            [&](const auto& array) -> void {
                const auto current = array.get_alloc_kind();
                if (first) {
                    common = current;
                    first = false;
                }
                else if (current != common) {
                    throw invalid_argument{
                        dal::detail::error_messages::alloc_kinds_of_arrays_do_not_match()
                    };
                }
            },
            arrays...);

        return common;
    }
}

template <typename Array>
struct array_type_map {};

template <typename Type>
struct array_type_map<array<Type>> {
    using type = Type;
};

template <typename Type>
struct array_type_map<array_impl<Type>> {
    using type = Type;
};

template <typename Type>
struct array_type_map<chunked_array<Type>> {
    using type = Type;
};

template <typename Array, typename Raw = std::decay_t<Array>>
using array_type_t = std::decay_t<typename array_type_map<Raw>::type>;

template <typename... Arrays>
inline table_metadata make_default_metadata_from_arrays(const Arrays&... arrays) {
    const alloc_kind kind = deduce_common_alloc_kind(arrays...);
    return make_default_metadata<array_type_t<Arrays>...>(kind);
}

} // namespace v1

using v1::find_array_dtypes;
using v1::find_array_ftypes;

using v1::make_default_metadata;
using v1::make_default_metadata_from_arrays;

} // namespace oneapi::dal::detail
