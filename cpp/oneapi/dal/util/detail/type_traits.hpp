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

#pragma once

#include <type_traits>

namespace oneapi::dal::detail {
namespace v1 {

#define ONEDAL_HAS_METHOD_TRAIT(return_value, method_name, params, checker_name)            \
    template <typename _Type>                                                               \
    struct has_method_##checker_name {                                                      \
    private:                                                                                \
        using pass = std::true_type;                                                        \
        using fail = std::false_type;                                                       \
                                                                                            \
        template <typename U, return_value(U::*f) params>                                   \
        struct checker_t {};                                                                \
                                                                                            \
        template <class U>                                                                  \
        static pass test(checker_t<U, &U::method_name> *);                                  \
        template <class U>                                                                  \
        static fail test(...);                                                              \
                                                                                            \
    public:                                                                                 \
        static constexpr bool value = std::is_same_v<pass, decltype(test<_Type>(nullptr))>; \
    };                                                                                      \
                                                                                            \
    template <typename _Type>                                                               \
    static inline constexpr bool has_method_##checker_name##_v =                            \
        has_method_##checker_name<_Type>::value;

#define ONEDAL_SIMPLE_HAS_METHOD_TRAIT(return_value, method_name, params) \
    ONEDAL_HAS_METHOD_TRAIT(return_value, method_name, params, method_name)

} // namespace v1

} // namespace oneapi::dal::detail
