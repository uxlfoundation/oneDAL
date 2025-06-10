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

#include "oneapi/dal/table/detail/table_iface.hpp"
#include "oneapi/dal/detail/serialization.hpp"

namespace oneapi::dal {

namespace detail {
namespace v1 {
class table_metadata_impl;
} // namespace v1

using v1::table_metadata_impl;

} // namespace detail

namespace v1 {

enum class feature_type { nominal, ordinal, interval, ratio };
enum class data_layout { unknown, row_major, column_major };
enum class sparse_indexing { zero_based, one_based };

class ONEDAL_EXPORT table_metadata {
    friend detail::pimpl_accessor;
    friend detail::serialization_accessor;

public:
    /// Creates the metadata instance without information about the features.
    /// The :literal:`feature_count` should be set to zero.
    /// The :literal:`data_type` and :literal:`feature_type` properties should not be initialized.
    table_metadata();

    /// Creates the metadata instance from external information about the data types and the
    /// feature types.
    /// @param dtypes The data types of the features. Assigned into the :literal:`data_type` property.
    /// @param ftypes The feature types. Assigned into the :literal:`feature_type` property.
    /// @pre :expr:`dtypes.get_count() == ftypes.get_count()`
    table_metadata(const dal::array<data_type>& dtypes, const dal::array<feature_type>& ftypes);

    /// The number of features that metadata contains information about
    /// @invariant :literal:`feature_count >= 0`
    std::int64_t get_feature_count() const;

    /// Feature types in the metadata object. Should be within the range ``[0, feature_count)``
    const feature_type& get_feature_type(std::int64_t feature_index) const;

    /// Data types of the features in the metadata object. Should be within the range ``[0, feature_count)``
    const data_type& get_data_type(std::int64_t feature_index) const;

    /// Get data types of features in bulk
    const dal::array<data_type>& get_data_types() const;

    /// Get feature types in bulk
    const dal::array<feature_type>& get_feature_types() const;

private:
    void serialize(detail::output_archive& ar) const;
    void deserialize(detail::input_archive& ar);

    detail::pimpl<detail::table_metadata_impl> impl_;
};

class ONEDAL_EXPORT table {
    friend detail::pimpl_accessor;
    friend detail::serialization_accessor;

public:
    /// An empty table constructor: creates the table instance with zero number of rows and columns.
    table();

    /// Creates a new table instance that shares the implementation with another one.
    table(const table&) = default;

    /// Creates a new table instance and moves implementation from another one into it.
    table(table&&);

    /// Replaces the implementation by another one.
    table& operator=(const table&) = default;

    /// Swaps the implementation of this object and another one.
    table& operator=(table&&);

    /// Destroys the table instance and releases the associated implementation.
    /// TODO: Enable ~table in next major release
    //virtual ~table() = default;

    /// Indicates whether a table contains non-zero number of rows and columns.
    bool has_data() const noexcept;

    /// The number of columns in the table.
    /// @remark default = 0
    std::int64_t get_column_count() const;

    /// The number of rows in the table.
    /// @remark default = 0
    std::int64_t get_row_count() const;

    /// The metadata object that holds additional information
    /// about the data within the table.
    /// @remark default = :expr:`table_metadata{}`
    const table_metadata& get_metadata() const;

    /// The runtime id of the table type.
    /// Each table sub-type has its unique ``kind``.
    /// An empty table has a unique ``kind`` value as well.
    std::int64_t get_kind() const;

    /// The layout of the data within the table
    /// @remark default = :expr:`data_layout::unknown`
    data_layout get_data_layout() const;

protected:
    explicit table(detail::table_iface* impl) : impl_(impl) {}
    explicit table(const detail::shared<detail::table_iface>& impl) : impl_(impl) {}

    static void validate_input_dimensions(std::int64_t row_count, std::int64_t column_count);

    void init_impl(detail::table_iface* impl) {
        impl_.reset(impl);
    }

private:
    void serialize(detail::output_archive& ar) const;
    void deserialize(detail::input_archive& ar);

    detail::pimpl<detail::table_iface> impl_;
};

} // namespace v1

using v1::feature_type;
using v1::data_layout;
using v1::sparse_indexing;
using v1::table_metadata;
using v1::table;

} // namespace oneapi::dal
