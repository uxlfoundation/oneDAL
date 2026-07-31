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

#include <utility>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/chunked_array.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/metadata_utils.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal {
namespace v1 {

class ONEDAL_EXPORT heterogen_table : public table {
    friend detail::pimpl_accessor;

public:
    static std::int64_t kind();

    /// Creates a new ``homogen_table`` instance with zero number of rows and columns.
    heterogen_table();

    /// Casts an object of the base table type to a heterogen table. If cast is
    /// not possible, the operation is equivalent to a default constructor call.
    explicit heterogen_table(const table& other);

    static heterogen_table empty(const table_metadata& meta);

    template <typename... Arrays>
    static heterogen_table wrap(Arrays&&... arrays) {
        using detail::integral_cast;

        auto meta = detail::make_default_metadata_from_arrays(arrays...);
        heterogen_table result = heterogen_table::empty(meta);

        [[maybe_unused]] const std::size_t ccount = sizeof...(Arrays);
        [[maybe_unused]] const auto count = integral_cast<std::int64_t>(ccount);

        ONEDAL_ASSERT(count == meta.get_feature_count());
        ONEDAL_ASSERT(count == result.get_column_count());

        std::int64_t column = 0l;

        detail::apply(
            [&](const auto& array) -> void {
                result.set_column(column++, array);
            },
            std::forward<Arrays>(arrays)...);

        ONEDAL_ASSERT(column == count);

        return result;
    }

    template <typename T>
    chunked_array<T> get_column(std::int64_t column) const {
        auto [dtype, arr] = this->get_column_impl(column);
        ONEDAL_ASSERT(dtype == detail::make_data_type<T>());
        return chunked_array<T>{ std::move(arr) };
    }

    template <typename T>
    heterogen_table& set_column(std::int64_t column, array<T> arr) {
        const auto as_chunked = chunked_array<T>{ std::move(arr) };
        const auto dt = this->get_metadata().get_data_type(column);
        this->set_column(column, dt, std::move(as_chunked));
        return *this;
    }

    template <typename T>
    heterogen_table& set_column(std::int64_t column, chunked_array<T> arr) {
        const auto dt = this->get_metadata().get_data_type(column);
        this->set_column_impl(column, dt, std::move(arr));
        return *this;
    }

    std::int64_t get_kind() const {
        return kind();
    }

private:
    void set_column_impl(std::int64_t column, data_type dt, detail::chunked_array_base arr);
    std::pair<data_type, detail::chunked_array_base> get_column_impl(std::int64_t column) const;

    explicit heterogen_table(detail::heterogen_table_iface* impl) : table(impl) {}
    explicit heterogen_table(const detail::shared<detail::heterogen_table_iface>& impl)
            : table(impl) {}
};

} // namespace v1

using v1::heterogen_table;

} // namespace oneapi::dal
