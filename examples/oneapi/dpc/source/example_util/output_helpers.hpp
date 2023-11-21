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

#include <iomanip>
#include <iostream>

#include "example_util/dpc_helpers.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/common.hpp"

template <typename Type>
std::ostream &print_on_host(std::ostream &stream, const oneapi::dal::array<Type> &array) {
    const std::int64_t count = array.get_count();

    if (count < std::int64_t(1)) {
        stream << "An empty array" << std::endl;
    }

    constexpr std::int32_t precision = std::is_floating_point_v<Type> ? 3 : 0;

    stream << std::setw(10);
    stream << std::setprecision(precision);
    stream << std::setiosflags(std::ios::fixed);
    for (std::int64_t i = 0l; i < count; ++i) {
        stream << array[i] << ' ';
    }

    return stream;
}

template <typename Type>
std::ostream &operator<<(std::ostream &stream, const oneapi::dal::array<Type> &array) {
    oneapi::dal::array<Type> array_on_host = to_host(array);
    return print_on_host(stream, array_on_host);
}

std::ostream &operator<<(std::ostream &stream, const oneapi::dal::table &table) {
    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    if (table.get_row_count() <= 10) {
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    else {
        for (std::int64_t i = 0; i < 5; i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
        std::cout << "..." << (table.get_row_count() - 10) << " lines skipped..." << std::endl;
        for (std::int64_t i = table.get_row_count() - 5; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    return stream;
}
