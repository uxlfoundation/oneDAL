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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/detail/array_utils.hpp"

namespace oneapi::dal {
namespace v1 {

class ONEDAL_EXPORT homogen_table : public table {
    friend detail::pimpl_accessor;

public:
    /// Returns the unique id of ``homogen_table`` class.
    static std::int64_t kind();

    /// Creates a new ``homogen_table`` instance from externally-defined data block. Table
    /// object refers to the data but does not own it. The responsibility to
    /// free the data remains on the user side.
    /// The :literal:`data` should point to the ``data_pointer`` memory block.
    ///
    /// @tparam Data        The type of elements in the data block that will be stored into the
    ///                     table. The table initializes data types of metadata with this data type.
    ///                     The feature types should be set to default values for :literal:`Data` type:
    ///                     contiguous for floating-point, ordinal for integer types. The :literal:`Data`
    ///                     type should be at least :expr:`float`, :expr:`double` or :expr:`std::int32_t`.
    ///
    /// @param data_pointer The pointer to a homogeneous data block.
    /// @param row_count    The number of rows in the table.
    /// @param column_count The number of columns in the table.
    /// @param layout       The layout of the data. Should be :literal:`data_layout::row_major` or
    ///                     :literal:`data_layout::column_major`.
    template <typename Data>
    static homogen_table wrap(const Data* data_pointer,
                              std::int64_t row_count,
                              std::int64_t column_count,
                              data_layout layout = data_layout::row_major) {
        return homogen_table{ data_pointer,
                              row_count,
                              column_count,
                              dal::detail::empty_delete<const Data>(),
                              layout };
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new ``homogen_table`` instance from externally-defined data block. Table
    /// object refers to the data but does not own it. The responsibility to
    /// free the data remains on the user side.
    /// The :literal:`data` should point to the ``data_pointer`` memory block.
    ///
    /// @tparam Data        The type of elements in the data block that will be stored into the
    ///                     table. The table initializes data types of metadata with this data type.
    ///                     The feature types should be set to default values for :literal:`Data` type:
    ///                     contiguous for floating-point, ordinal for integer types. The :literal:`Data`
    ///                     type should be at least :expr:`float`, :expr:`double` or :expr:`std::int32_t`.
    ///
    /// @param queue        The SYCL* queue object
    /// @param data_pointer The pointer to a homogeneous data block.
    /// @param row_count    The number of rows in the table.
    /// @param column_count The number of columns in the table.
    /// @param dependencies Events indicating availability of the :literal:`Data` for reading or writing.
    /// @param layout       The layout of the data. Should be :literal:`data_layout::row_major` or
    ///                     :literal:`data_layout::column_major`.
    template <typename Data>
    static homogen_table wrap(const sycl::queue& queue,
                              const Data* data_pointer,
                              std::int64_t row_count,
                              std::int64_t column_count,
                              const std::vector<sycl::event>& dependencies = {},
                              data_layout layout = data_layout::row_major) {
        return homogen_table{ queue,
                              data_pointer,
                              row_count,
                              column_count,
                              dal::detail::empty_delete<const Data>(),
                              dependencies,
                              layout };
    }
#endif

    /// Creates a new ``homogen_table`` instance from an array.
    /// The created table shares data ownership with the given array.
    ///
    /// @tparam Data        The type of elements in the data block that will be stored into the
    ///                     table. The table initializes data types of metadata with this data type.
    ///                     The feature types should be set to default values for :literal:`Data` type:
    ///                     contiguous for floating-point, ordinal for integer types. The :literal:`Data`
    ///                     type should be at least :expr:`float`, :expr:`double` or :expr:`std::int32_t`.
    ///
    /// @param data         The array that stores a homogeneous data block.
    /// @param row_count    The number of rows in the table.
    /// @param column_count The number of columns in the table.
    /// @param layout       The layout of the data. Should be :literal:`data_layout::row_major` or
    ///                     :literal:`data_layout::column_major`.
    template <typename Data>
    static homogen_table wrap(const dal::array<Data>& data,
                              std::int64_t row_count,
                              std::int64_t column_count,
                              data_layout layout = data_layout::row_major) {
        return homogen_table{ data, row_count, column_count, layout };
    }

    /// Creates a new ``homogen_table`` instance with zero number of rows and columns.
    homogen_table();

    /// Casts an object of the base table type to a homogen table. If cast is
    /// not possible, the operation is equivalent to a default constructor call.
    explicit homogen_table(const table& other);

    /// Creates a new ``homogen_table`` instance from externally-defined data block.
    /// Table object owns the data pointer.
    /// The :literal:`data` should point to the ``data_pointer`` memory block.
    ///
    /// @tparam Data         The type of elements in the data block that will be stored into the table.
    ///                      The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                      or :expr:`std::int32_t`.
    /// @tparam ConstDeleter The type of a deleter called on ``data_pointer`` when
    ///                      the last table that refers it is out of the scope.
    ///
    /// @param data_pointer  The pointer to a homogeneous data block.
    /// @param row_count     The number of rows in the table.
    /// @param column_count  The number of columns in the table.
    /// @param data_deleter  The deleter that is called on the ``data_pointer`` when the last table that refers it
    ///                      is out of the scope.
    /// @param layout        The layout of the data. Should be :literal:`data_layout::row_major` or
    ///                      :literal:`data_layout::column_major`.
    template <typename Data, typename ConstDeleter>
    homogen_table(const Data* data_pointer,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  ConstDeleter&& data_deleter,
                  data_layout layout = data_layout::row_major) {
        init_impl(detail::default_host_policy{},
                  row_count,
                  column_count,
                  data_pointer,
                  std::forward<ConstDeleter>(data_deleter),
                  layout);
    }

#ifdef ONEDAL_DATA_PARALLEL
    /// Creates a new ``homogen_table`` instance from externally-defined data block.
    /// Table object owns the data pointer.
    /// The :literal:`data` should point to the ``data_pointer`` memory block.
    ///
    /// @tparam Data         The type of elements in the data block that will be stored into the table.
    ///                      The :literal:`Data` type should be at least :expr:`float`, :expr:`double`
    ///                      or :expr:`std::int32_t`.
    /// @tparam ConstDeleter The type of a deleter called on ``data_pointer`` when
    ///                      the last table that refers it is out of the scope.
    ///
    /// @param queue         The SYCL* queue object
    /// @param data_pointer  The pointer to a homogeneous data block.
    /// @param row_count     The number of rows in the table.
    /// @param column_count  The number of columns in the table.
    /// @param data_deleter  The deleter that is called on the ``data_pointer`` when the last table that refers it
    ///                      is out of the scope.
    /// @param dependencies  Events indicating availability of the :literal:`Data` for reading or writing.
    /// @param layout        The layout of the data. Should be :literal:`data_layout::row_major` or
    ///                      :literal:`data_layout::column_major`.
    template <typename Data, typename ConstDeleter>
    homogen_table(const sycl::queue& queue,
                  const Data* data_pointer,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  ConstDeleter&& data_deleter,
                  const std::vector<sycl::event>& dependencies = {},
                  data_layout layout = data_layout::row_major) {
        init_impl(detail::data_parallel_policy{ queue },
                  row_count,
                  column_count,
                  data_pointer,
                  std::forward<ConstDeleter>(data_deleter),
                  layout);
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    /// Returns the :literal:`data` pointer cast to the :literal:`Data` type. No checks are
    /// performed that this type is the actual type of the data within the table. If table
    /// has no data, returns :literal:`nullptr`.
    template <typename Data>
    const Data* get_data() const {
        return reinterpret_cast<const Data*>(this->get_data());
    }

    /// The pointer to the data block within the table.
    /// Should be equal to ``nullptr`` when :expr:`row_count == 0` and :expr:`column_count == 0`.
    const void* get_data() const;

    /// The unique id of the homogen table type.
    std::int64_t get_kind() const {
        return kind();
    }

private:
    template <typename Data>
    homogen_table(const dal::array<Data>& data,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  data_layout layout = data_layout::row_major) {
        init_impl(data, row_count, column_count, layout);
    }

    explicit homogen_table(detail::homogen_table_iface* impl) : table(impl) {}
    explicit homogen_table(const detail::shared<detail::homogen_table_iface>& impl) : table(impl) {}

    template <typename Policy, typename Data, typename ConstDeleter>
    void init_impl(const Policy& policy,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const Data* data_pointer,
                   ConstDeleter&& data_deleter,
                   data_layout layout) {
        validate_input_dimensions(row_count, column_count);

        const auto data = detail::array_via_policy<Data>::wrap(
            policy,
            data_pointer,
            detail::check_mul_overflow(row_count, column_count),
            std::forward<ConstDeleter>(data_deleter));

        init_impl(policy,
                  row_count,
                  column_count,
                  detail::reinterpret_array_cast<byte_t>(data),
                  detail::make_data_type<Data>(),
                  layout);
    }

    template <typename Data>
    void init_impl(const dal::array<Data>& data,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   data_layout layout) {
        validate_input_dimensions(row_count, column_count);

        if (data.get_count() < detail::check_mul_overflow(row_count, column_count)) {
            using msg = detail::error_messages;
            throw invalid_argument{ msg::rc_and_cc_do_not_match_element_count_in_array() };
        }

        detail::dispatch_by_policy(data, [&](auto policy) {
            init_impl(policy,
                      row_count,
                      column_count,
                      detail::reinterpret_array_cast<byte_t>(data),
                      detail::make_data_type<Data>(),
                      layout);
        });
    }

    template <typename Policy>
    void init_impl(const Policy& policy,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const dal::array<byte_t>& data,
                   const data_type& dtype,
                   data_layout layout);
};

} // namespace v1

using v1::homogen_table;

} // namespace oneapi::dal
