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

#include <variant>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/table_utils.hpp"
#include "oneapi/dal/table/row_accessor.hpp"


namespace oneapi::dal::backend::primitives {


/// Convert a table to a 2D ndarray
///
/// @tparam Type The type of the memory block elements within the ndarray.
///
/// @param table Input table
///
/// @return The new 2D ndarray instance.
template <typename Type>
inline ndarray<Type, 2> table2ndarray(const table& table) {
    ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS(service::table2ndarray,
                                           table.get_row_count(),
                                           table.get_column_count());
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull({ 0, -1 });
    return ndarray<Type, 2>::wrap(data, { table.get_row_count(), table.get_column_count() });
}

/// Convert a table to a 1D ndarray
///
/// @tparam Type The type of the memory block elements within the ndarray.
///
/// @param table Input table
///
/// @return The new 1D ndarray instance.
template <typename Type>
inline ndarray<Type, 1> table2ndarray_1d(const table& table) {
    ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS(service::table2ndarray_1d,
                                           table.get_row_count(),
                                           table.get_column_count());
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull({ 0, -1 });
    return ndarray<Type, 1>::wrap(data, { data.get_count() });
}

#ifdef ONEDAL_DATA_PARALLEL

/// Check if a USM pointer is accessible on the target device with the requested
/// allocation kind. Returns true if no copy is needed.
///
/// @tparam Type The element type (used for pointer casting).
///
/// @param q     The SYCL* queue.
/// @param ptr   The USM pointer to check.
/// @param alloc The requested USM allocation type.
///
/// @return True if the pointer is already suitable for the requested alloc on the queue's device.
template <typename Type>
inline bool is_ptr_accessible(sycl::queue& q, const Type* ptr, sycl::usm::alloc alloc) {
    const auto context = q.get_context();
    const auto ptr_alloc = sycl::get_pointer_type(ptr, context);
    if (ptr_alloc != alloc) {
        return false;
    }
    if (ptr_alloc == sycl::usm::alloc::device) {
        return sycl::get_pointer_device(ptr, context) == q.get_device();
    }
    return true;
}

/// Transpose a 2D ndarray from one order to another.
///
/// @tparam Type      The element type.
/// @tparam src_order The source ndorder.
/// @tparam dst_order The destination ndorder.
///
/// @param q    The SYCL* queue.
/// @param src  The source ndview.
/// @param dst  The destination ndview (must be pre-allocated).
///
/// @return Event indicating completion of the transpose.
template <typename Type, ndorder src_order, ndorder dst_order>
inline sycl::event transpose_copy(sycl::queue& q,
                                  const ndview<Type, 2, src_order>& src,
                                  ndview<Type, 2, dst_order>& dst,
                                  const event_vector& deps = {}) {
    return copy(q, dst, src, deps);
}

/// Get the raw data pointer from a homogen_table and ensure it is accessible
/// with the requested allocation kind. The requested ndorder must match the
/// table's actual data layout (no transpose is performed).
/// Returns an ndarray that either wraps the original pointer (zero-copy) or
/// holds a flat copy in newly allocated memory.
///
/// @tparam Type  The element type.
/// @tparam order The ndorder matching the table's data layout.
///
/// @param q     The SYCL* queue.
/// @param table Input homogen_table.
/// @param alloc The requested USM allocation type.
///
/// @return A 2D ndarray over the table's data in the specified order.
template <typename Type, ndorder order>
inline ndarray<Type, 2, order> homogen_table_to_same_order_ndarray(sycl::queue& q,
                                                                   const table& table,
                                                                   sycl::usm::alloc alloc) {
    using arr_t = ndarray<Type, 2, order>;
    const auto row_count = table.get_row_count();
    const auto column_count = table.get_column_count();
    ONEDAL_ASSERT(table.get_kind() == homogen_table::kind());

    [[maybe_unused]] constexpr auto expected_layout =
        (order == ndorder::c) ? data_layout::row_major : data_layout::column_major;
    ONEDAL_ASSERT(table.get_data_layout() == expected_layout);

    const auto iface = dal::detail::get_homogen_table_iface(table);
    const auto byte_data = iface->get_data();
    const std::int64_t count = row_count * column_count;
    const Type* ptr = reinterpret_cast<const Type*>(byte_data.get_data());

    const auto context = q.get_context();
    const auto src_alloc = sycl::get_pointer_type(ptr, context);

    // Mirror the logic from pull_rows_impl: determine if a copy is needed
    // based on source and destination allocation kinds.
    // alloc_kind_requires_copy semantics:
    //   device requested: copy needed from host or usm_host
    //   shared requested: copy needed unless source is also shared
    //   host/usm_host requested: copy needed only from device
    const bool nocopy = [&]() {
        switch (alloc) {
            case sycl::usm::alloc::device:
                return (src_alloc == sycl::usm::alloc::device ||
                        src_alloc == sycl::usm::alloc::shared);
            case sycl::usm::alloc::shared:
                return (src_alloc == sycl::usm::alloc::shared);
            case sycl::usm::alloc::host:
                return (src_alloc == sycl::usm::alloc::host ||
                        src_alloc == sycl::usm::alloc::unknown);
            default: return false;
        }
    }();

    if (nocopy) {
        auto owned_data = dal::array<Type>{ byte_data, ptr, count };
        return arr_t::wrap(owned_data, { row_count, column_count });
    }

    auto result = arr_t::empty(q, { row_count, column_count }, alloc);
    dal::backend::copy(q, result.get_mutable_data(), ptr, count).wait_and_throw();
    return result;
}

/// Convert a table to a 2D ndarray with row-major data order.
/// For homogeneous tables:
///   - row-major + same alloc: zero-copy wrap of raw pointer
///   - row-major + different alloc: flat copy to target device
///   - column-major: transpose via single kernel
/// For heterogeneous tables: uses row_accessor which handles per-column type conversion.
///
/// @tparam Type The type of the memory block elements within the ndarray.
///
/// @param q     The SYCL* queue.
/// @param table Input table.
/// @param alloc The USM allocation type.
///
/// @return The new 2D ndarray instance with row-major order.
template <typename Type>
inline ndarray<Type, 2, ndorder::c> table2ndarray_rm(sycl::queue& q,
                                                     const table& table,
                                                     sycl::usm::alloc alloc) {
    using rm_t = ndarray<Type, 2, ndorder::c>;
    const auto row_count = table.get_row_count();
    const auto column_count = table.get_column_count();

    if (table.get_kind() == homogen_table::kind() &&
        table.get_metadata().get_data_type(0) == dal::detail::make_data_type<Type>()) {
        const auto layout = table.get_data_layout();

        if (layout == data_layout::row_major) {
            return homogen_table_to_same_order_ndarray<Type, ndorder::c>(q, table, alloc);
        }

        if (layout == data_layout::column_major) {
            auto src =
                homogen_table_to_same_order_ndarray<Type, ndorder::f>(q, table, alloc);
            auto dst = rm_t::empty(q, { row_count, column_count }, alloc);
            transpose_copy(q, src, dst).wait_and_throw();
            return dst;
        }
    }

    // Fallback: row_accessor handles type conversion and heterogeneous tables
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull(q, { 0, -1 }, alloc);
    return rm_t::wrap(data, { row_count, column_count });
}

/// Convert a table to a 2D ndarray with column-major data order.
/// For homogeneous tables:
///   - column-major + same alloc: zero-copy wrap of raw pointer
///   - column-major + different alloc: flat copy to target device
///   - row-major: transpose via single kernel
/// For heterogeneous tables: uses row_accessor then transposes.
///
/// @tparam Type The type of the memory block elements within the ndarray.
///
/// @param q     The SYCL* queue.
/// @param table Input table.
/// @param alloc The USM allocation type.
///
/// @return The new 2D ndarray instance with column-major order.
template <typename Type>
inline ndarray<Type, 2, ndorder::f> table2ndarray_cm(sycl::queue& q,
                                                     const table& table,
                                                     sycl::usm::alloc alloc) {
    using cm_t = ndarray<Type, 2, ndorder::f>;
    const auto row_count = table.get_row_count();
    const auto column_count = table.get_column_count();

    if (table.get_kind() == homogen_table::kind() &&
        table.get_metadata().get_data_type(0) == dal::detail::make_data_type<Type>()) {
        const auto layout = table.get_data_layout();

        if (layout == data_layout::column_major) {
            return homogen_table_to_same_order_ndarray<Type, ndorder::f>(q, table, alloc);
        }

        if (layout == data_layout::row_major) {
            auto src =
                homogen_table_to_same_order_ndarray<Type, ndorder::c>(q, table, alloc);
            auto dst = cm_t::empty(q, { row_count, column_count }, alloc);
            transpose_copy(q, src, dst).wait_and_throw();
            return dst;
        }
    }

    // Fallback: row_accessor handles type conversion and heterogeneous tables
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull(q, { 0, -1 }, alloc);
    auto src = ndarray<Type, 2, ndorder::c>::wrap(data, { row_count, column_count });
    auto dst = cm_t::empty(q, { row_count, column_count }, alloc);
    transpose_copy(q, src, dst).wait_and_throw();
    return dst;
}

/// Convert a table to a 2D ndarray
///
/// @tparam Type  The type of the memory block elements within the ndarray.
/// @tparam order The order of the ndarray.
///
/// @param q     The SYCL* queue.
/// @param table Input table.
/// @param alloc The USM allocation type.
///
/// @return The new 2D ndarray instance.
template <typename Type, ndorder order = ndorder::c>
inline ndarray<Type, 2, order> table2ndarray(sycl::queue& q,
                                             const table& table,
                                             sycl::usm::alloc alloc = sycl::usm::alloc::shared) {
    ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS_QUEUE(service::table2ndarray_queue,
                                                 q,
                                                 table.get_row_count());
    // std::cerr << "table2ndarray sizeof:" << dal::detail::get_data_type_size(table.get_metadata().get_data_type(0)) << " "
    // << " sizeof(Type)=" << sizeof(Type) << std::endl;

    if constexpr (order == ndorder::c) {
        return table2ndarray_rm<Type>(q, table, alloc);
    }
    else {
        return table2ndarray_cm<Type>(q, table, alloc);
    }
}

/// Convert a table to a 2D ndarray with the data order determined by the table layout.
///
/// @tparam Type  The type of the memory block elements within the ndarray.
///
/// @param q     The SYCL* queue.
/// @param table Input table.
/// @param alloc The USM allocation type.
///
/// @return The new 2D ndarray instance. If the table layout is row-major, the ndarray will have
///         row-major data order. If the table layout is column-major, the ndarray will have
///         column-major data order.
template <typename Type>
inline auto table2ndarray_variant(sycl::queue& q, const table& table, sycl::usm::alloc alloc) {
    ONEDAL_ASSERT(table.has_data());
    const auto data_layout = table.get_data_layout();
    using var1_t = ndarray<Type, 2, ndorder::c>;
    using var2_t = ndarray<Type, 2, ndorder::f>;
    std::variant<var1_t, var2_t> result;
    if (data_layout == decltype(data_layout)::row_major) {
        result = table2ndarray_rm<Type>(q, table, alloc);
    }
    if (data_layout == decltype(data_layout)::column_major) {
        result = table2ndarray_cm<Type>(q, table, alloc);
    }
    return result;
}

/// Convert a table to a 1D ndarray with row-major element order.
/// For homogeneous tables: uses optimized path via table2ndarray_rm
/// (zero-copy for row-major on device, transpose for column-major).
/// For heterogeneous tables: falls back to row_accessor.
///
/// @tparam Type The type of the memory block elements within the ndarray.
///
/// @param q     The SYCL* queue.
/// @param table Input table.
/// @param alloc The USM allocation type.
///
/// @return The new 1D ndarray instance.
template <typename Type>
inline ndarray<Type, 1> table2ndarray_1d(sycl::queue& q,
                                         const table& table,
                                         sycl::usm::alloc alloc = sycl::usm::alloc::shared) {
    const auto rm = table2ndarray_rm<Type>(q, table, alloc);
    return ndarray<Type, 1>::wrap(rm.flatten(q), { rm.get_count() });
}

/// Flatten a table into a 1D ndarray preserving the raw storage order.
/// No transpose is performed — the element order depends on the table's
/// data layout (row-major yields row-wise, column-major yields column-wise).
/// This is suitable for element-wise operations (e.g. finiteness checks)
/// where traversal order does not matter.
/// Supports both homogeneous and heterogeneous tables.
///
/// @tparam Type The type of the memory block elements within the ndarray.
///
/// @param q     The SYCL* queue.
/// @param table Input table.
/// @param alloc The requested USM allocation type.
///
/// @return A 1D ndarray over the table's data.
template <typename Type>
inline ndarray<Type, 1> flatten_table_by_layout(sycl::queue& q,
                                                const table& table,
                                                sycl::usm::alloc alloc = sycl::usm::alloc::device) {
    const auto layout = table.get_data_layout();
    if (layout == data_layout::column_major) {
        const auto cm = table2ndarray_cm<Type>(q, table, alloc);
        return ndarray<Type, 1>::wrap(cm.flatten(q), { cm.get_count() });
    }
    const auto rm = table2ndarray_rm<Type>(q, table, alloc);
    return ndarray<Type, 1>::wrap(rm.flatten(q), { rm.get_count() });
}
#endif

} // namespace oneapi::dal::backend::primitives
