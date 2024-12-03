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

#include "oneapi/dal/table/backend/convert.hpp"

#include <algorithm>
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/interop/data_conversion.hpp"

namespace oneapi::dal::backend {

static void convert_vector(const void* src,
                           void* dst,
                           data_type src_type,
                           data_type dst_type,
                           std::int64_t src_stride,
                           std::int64_t dst_stride,
                           std::int64_t element_count) {
    if (src_stride == 1 && dst_stride == 1) {
        interop::daal_convert(src, dst, src_type, dst_type, element_count);
    }
    else {
        const std::int64_t src_element_size = dal::detail::get_data_type_size(src_type);
        const std::int64_t dst_element_size = dal::detail::get_data_type_size(dst_type);
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, src_stride, src_element_size);
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, dst_stride, dst_element_size);
        interop::daal_convert(src,
                              dst,
                              src_type,
                              dst_type,
                              src_stride * src_element_size,
                              dst_stride * dst_element_size,
                              element_count);
    }
}

void convert_vector(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t element_count) {
    convert_vector(src, dst, src_type, dst_type, 1, 1, element_count);
}

void convert_vector(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t src_stride,
                    std::int64_t dst_stride,
                    std::int64_t element_count) {
    if (src_stride == 1 && dst_stride == 1) {
        interop::daal_convert(src, dst, src_type, dst_type, element_count);
    }
    else {
        const std::int64_t src_element_size = dal::detail::get_data_type_size(src_type);
        const std::int64_t dst_element_size = dal::detail::get_data_type_size(dst_type);
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, src_stride, src_element_size);
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, dst_stride, dst_element_size);
        interop::daal_convert(src,
                              dst,
                              src_type,
                              dst_type,
                              src_stride * src_element_size,
                              dst_stride * dst_element_size,
                              element_count);
    }
}

void convert_matrix(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    const std::int64_t src_row_stride,
                    const std::int64_t dst_row_stride,
                    const std::int64_t src_col_stride,
                    const std::int64_t dst_col_stride,
                    const std::int64_t dst_row_count,
                    const std::int64_t dst_col_count) {
    dispatch_by_data_type(src_type, [&](auto src_type_id) {
        dispatch_by_data_type(dst_type, [&](auto dst_type_id) {
            using src_t = decltype(src_type_id);
            using dst_t = decltype(dst_type_id);
            auto src_ptr = static_cast<const src_t*>(src);
            auto dst_ptr = static_cast<dst_t*>(dst);
            for (std::int64_t i = 0; i < dst_row_count; i++) {
                backend::convert_vector(policy,
                                        src_ptr + i * src_row_stride,
                                        dst_ptr + i * dst_row_stride,
                                        src_type,
                                        dst_type,
                                        src_col_stride,
                                        dst_col_stride,
                                        dst_col_count);
            }
        });
    });
}

template <typename DataType>
void shift_array_values(DataType* arr, const std::int64_t element_count, const DataType shift) {
    if (shift == DataType(0))
        return;

    for (std::int64_t i = 0; i < element_count; ++i)
        arr[i] += shift;
}

void shift_array_values(const detail::default_host_policy& policy,
                        void* arr,
                        data_type arr_type,
                        const std::int64_t element_count,
                        const void* shift) {
    ONEDAL_ASSERT(arr);
    ONEDAL_ASSERT(shift);
    ONEDAL_ASSERT(element_count > 0);
    dispatch_by_data_type(arr_type, [&](auto arr_type_id) {
        using data_t = decltype(arr_type_id);
        shift_array_values<data_t>(static_cast<data_t*>(arr),
                                   element_count,
                                   static_cast<const data_t*>(shift)[0]);
    });
}

#ifdef ONEDAL_DATA_PARALLEL

template <typename Src, typename Dst>
static sycl::event convert_vector_kernel(sycl::queue& q,
                                         const Src* src,
                                         Dst* dst,
                                         std::int64_t src_stride,
                                         std::int64_t dst_stride,
                                         std::int64_t element_count,
                                         const event_vector& deps = {}) {
    const int src_stride_int = dal::detail::integral_cast<int>(src_stride);
    const int dst_stride_int = dal::detail::integral_cast<int>(dst_stride);
    const int element_count_int = dal::detail::integral_cast<int>(element_count);

    const std::int64_t required_local_size = 256;
    const std::int64_t local_size = std::min(down_pow2(element_count), required_local_size);
    const auto range = make_multiple_nd_range_1d(element_count, local_size);

    if (src_stride == 1 && dst_stride == 1) {
        return q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
                const int i = id.get_global_id();
                if (i < element_count_int) {
                    dst[i] = src[i];
                }
            });
        });
    }

    else {
        return q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
                const int i = id.get_global_id();
                if (i < element_count_int) {
                    const int src_i = i * src_stride_int;
                    const int dst_i = i * dst_stride_int;
                    dst[dst_i] = src[src_i];
                }
            });
        });
    }
}

sycl::event convert_vector_device2device(sycl::queue& q,
                                         const void* src,
                                         void* dst,
                                         data_type src_type,
                                         data_type dst_type,
                                         std::int64_t src_stride,
                                         std::int64_t dst_stride,
                                         std::int64_t element_count,
                                         const event_vector& deps) {
    return dispatch_by_data_type(src_type, [&](auto src_type_id) {
        return dispatch_by_data_type(dst_type, [&](auto dst_type_id) {
            using src_t = decltype(src_type_id);
            using dst_t = decltype(dst_type_id);
            return convert_vector_kernel<src_t, dst_t>(q,
                                                       static_cast<const src_t*>(src),
                                                       static_cast<dst_t*>(dst),
                                                       src_stride,
                                                       dst_stride,
                                                       element_count,
                                                       deps);
        });
    });
}

sycl::event convert_vector_device2host(sycl::queue& q,
                                       const void* src_device,
                                       void* dst_host,
                                       data_type src_type,
                                       data_type dst_type,
                                       std::int64_t src_stride,
                                       std::int64_t dst_stride,
                                       std::int64_t element_count,
                                       const event_vector& deps) {
    ONEDAL_ASSERT(src_device);
    ONEDAL_ASSERT(dst_host);
    ONEDAL_ASSERT(src_stride > 0);
    ONEDAL_ASSERT(dst_stride > 0);
    ONEDAL_ASSERT(element_count >= 0);
    ONEDAL_ASSERT(is_known_usm(q, src_device));

    // To perform conversion, we gather data from device to host in temporary
    // contigious array and then run host conversion function

    const std::int64_t dst_element_size_in_bytes = dal::detail::get_data_type_size(dst_type);
    const std::int64_t dst_size_in_bytes =
        dal::detail::check_mul_overflow(dst_element_size_in_bytes, element_count);
    const std::int64_t dst_stride_in_bytes =
        dal::detail::check_mul_overflow(dst_element_size_in_bytes, dst_stride);

    const std::int64_t src_element_size_in_bytes = dal::detail::get_data_type_size(src_type);
    const std::int64_t src_size_in_bytes =
        dal::detail::check_mul_overflow(src_element_size_in_bytes, element_count);
    const std::int64_t src_stride_in_bytes =
        dal::detail::check_mul_overflow(src_element_size_in_bytes, src_stride);
    if (src_type == dst_type && src_size_in_bytes == dst_size_in_bytes &&
        src_stride_in_bytes == dst_stride_in_bytes) {
        auto copy_event = memcpy_usm2host(q,
                                          dst_host,
                                          src_device,
                                          src_element_size_in_bytes * element_count,
                                          deps);

        return copy_event;
    }
    else {
        const auto tmp_host_unique = make_unique_usm_host(q, src_size_in_bytes);

        auto gather_event = gather_device2host(q,
                                               tmp_host_unique.get(),
                                               src_device,
                                               element_count,
                                               src_stride_in_bytes,
                                               src_element_size_in_bytes,
                                               deps);
        gather_event.wait_and_throw();

        convert_vector(dal::detail::default_host_policy{},
                       tmp_host_unique.get(),
                       dst_host,
                       src_type,
                       dst_type,
                       1L,
                       dst_stride,
                       element_count);

        return sycl::event{};
    }
}

sycl::event convert_vector_host2device(sycl::queue& q,
                                       const void* src_host,
                                       void* dst_device,
                                       data_type src_type,
                                       data_type dst_type,
                                       std::int64_t src_stride,
                                       std::int64_t dst_stride,
                                       std::int64_t element_count,
                                       const std::vector<sycl::event>& deps) {
    ONEDAL_ASSERT(src_host);
    ONEDAL_ASSERT(dst_device);
    ONEDAL_ASSERT(src_stride > 0);
    ONEDAL_ASSERT(dst_stride > 0);
    ONEDAL_ASSERT(element_count >= 0);
    ONEDAL_ASSERT(is_known_usm(q, dst_device));

    // To perform conversion, we perform conversion on the host and gather data
    // in temporary contigious array and then scatter it from host to device

    const std::int64_t dst_element_size_in_bytes = dal::detail::get_data_type_size(dst_type);
    const std::int64_t dst_size_in_bytes =
        dal::detail::check_mul_overflow(dst_element_size_in_bytes, element_count);
    const std::int64_t dst_stride_in_bytes =
        dal::detail::check_mul_overflow(dst_element_size_in_bytes, dst_stride);

    const std::int64_t src_element_size_in_bytes = dal::detail::get_data_type_size(src_type);
    const std::int64_t src_size_in_bytes =
        dal::detail::check_mul_overflow(src_element_size_in_bytes, element_count);
    const std::int64_t src_stride_in_bytes =
        dal::detail::check_mul_overflow(src_element_size_in_bytes, src_stride);
    if (src_type == dst_type && src_size_in_bytes == dst_size_in_bytes &&
        src_stride_in_bytes == dst_stride_in_bytes) {
        auto copy_event = memcpy_host2usm(q,
                                          dst_device,
                                          src_host,
                                          src_element_size_in_bytes * element_count,
                                          deps);

        return copy_event;
    }
    else {
        const auto tmp_host_unique = make_unique_usm_host(q, dst_size_in_bytes);

        convert_vector(dal::detail::default_host_policy{},
                       src_host,
                       tmp_host_unique.get(),
                       src_type,
                       dst_type,
                       src_stride,
                       1L,
                       element_count);
        const std::int64_t max_loop_range = std::numeric_limits<std::int32_t>::max();
        sycl::event scatter_event;
        if (element_count > max_loop_range) {
            scatter_event = scatter_host2device_blocking(q,
                                                         dst_device,
                                                         tmp_host_unique.get(),
                                                         element_count,
                                                         dst_stride_in_bytes,
                                                         dst_element_size_in_bytes,
                                                         deps);
        }
        else {
            scatter_event = scatter_host2device(q,
                                                dst_device,
                                                tmp_host_unique.get(),
                                                element_count,
                                                dst_stride_in_bytes,
                                                dst_element_size_in_bytes,
                                                deps);
        }
        return scatter_event;
    }
}

void convert_vector(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t element_count) {
    convert_vector(policy, src, dst, src_type, dst_type, 1, 1, element_count);
}

void convert_vector(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t src_stride,
                    std::int64_t dst_stride,
                    std::int64_t element_count) {
    // We treat shared memory as device assuming actual copy of shared memory
    // tend to reside on device
    sycl::queue& q = policy.get_queue();
    const bool src_device_friendly = is_device_friendly_usm(q, src);
    const bool dst_device_friendly = is_device_friendly_usm(q, dst);

    if (src_device_friendly && dst_device_friendly) {
        // Device -> Device
        convert_vector_device2device(q,
                                     src,
                                     dst,
                                     src_type,
                                     dst_type,
                                     src_stride,
                                     dst_stride,
                                     element_count)
            .wait_and_throw();
    }
    else if (src_device_friendly) {
        // Device -> Host
        convert_vector_device2host(q,
                                   src,
                                   dst,
                                   src_type,
                                   dst_type,
                                   src_stride,
                                   dst_stride,
                                   element_count)
            .wait_and_throw();
    }
    else if (dst_device_friendly) {
        // Host -> Device
        convert_vector_host2device(q,
                                   src,
                                   dst,
                                   src_type,
                                   dst_type,
                                   src_stride,
                                   dst_stride,
                                   element_count)
            .wait_and_throw();
    }
    else {
        // Host -> Host
        convert_vector(detail::default_host_policy{},
                       src,
                       dst,
                       src_type,
                       dst_type,
                       src_stride,
                       dst_stride,
                       element_count);
    }
}

template <typename Src, typename Dst>
sycl::event convert_matrix_host2device(sycl::queue& q,
                                       const Src* src_host,
                                       Dst* dst_device,
                                       data_type src_type,
                                       data_type dst_type,
                                       const std::int64_t src_row_stride,
                                       const std::int64_t dst_row_stride,
                                       const std::int64_t src_col_stride,
                                       const std::int64_t dst_col_stride,
                                       const std::int64_t dst_row_count,
                                       const std::int64_t dst_col_count) {
    ONEDAL_ASSERT(src_host);
    ONEDAL_ASSERT(dst_device);
    ONEDAL_ASSERT(src_row_stride > 0);
    ONEDAL_ASSERT(dst_row_stride > 0);
    ONEDAL_ASSERT(src_col_stride > 0);
    ONEDAL_ASSERT(dst_col_stride > 0);
    ONEDAL_ASSERT(dst_row_count >= 0);
    ONEDAL_ASSERT(dst_col_count >= 0);
    ONEDAL_ASSERT(is_known_usm(q, dst_device));

    const std::int64_t element_size_in_bytes = dal::detail::get_data_type_size(dst_type);
    const std::int64_t dst_count = dal::detail::check_mul_overflow(dst_col_count, dst_row_count);
    const std::int64_t dst_size_in_bytes =
        dal::detail::check_mul_overflow(element_size_in_bytes, dst_count);

    const auto tmp_host_unique = make_unique_usm_host<Dst>(q, dst_count);
    for (std::int64_t i = 0; i < dst_row_count; i++) {
        backend::convert_vector(detail::default_host_policy{},
                                src_host + i * src_row_stride,
                                tmp_host_unique.get() + i * dst_row_stride,
                                src_type,
                                dst_type,
                                src_col_stride,
                                dst_col_stride,
                                dst_col_count);
    }
    auto copy_event = memcpy(q, dst_device, tmp_host_unique.get(), dst_size_in_bytes);
    return copy_event;
}

void convert_matrix(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    const std::int64_t src_row_stride,
                    const std::int64_t dst_row_stride,
                    const std::int64_t src_col_stride,
                    const std::int64_t dst_col_stride,
                    const std::int64_t dst_row_count,
                    const std::int64_t dst_col_count) {
    dispatch_by_data_type(src_type, [&](auto src_type_id) {
        dispatch_by_data_type(dst_type, [&](auto dst_type_id) {
            using src_t = decltype(src_type_id);
            using dst_t = decltype(dst_type_id);
            auto src_ptr = static_cast<const src_t*>(src);
            auto dst_ptr = static_cast<dst_t*>(dst);
            sycl::queue& q = policy.get_queue();
            const bool src_device_friendly = is_device_friendly_usm(q, src_ptr);
            const bool dst_device_friendly = is_device_friendly_usm(q, dst_ptr);
            if (dst_device_friendly && !src_device_friendly) {
                convert_matrix_host2device<src_t, dst_t>(q,
                                                         src_ptr,
                                                         dst_ptr,
                                                         src_type,
                                                         dst_type,
                                                         src_row_stride,
                                                         dst_row_stride,
                                                         src_col_stride,
                                                         dst_col_stride,
                                                         dst_row_count,
                                                         dst_col_count)
                    .wait_and_throw();
            }
            else {
                for (std::int64_t i = 0; i < dst_row_count; i++) {
                    backend::convert_vector(policy,
                                            src_ptr + i * src_row_stride,
                                            dst_ptr + i * dst_row_stride,
                                            src_type,
                                            dst_type,
                                            src_col_stride,
                                            dst_col_stride,
                                            dst_col_count);
                }
            }
        });
    });
}

template <typename T>
sycl::event shift_array_values_device(sycl::queue& q,
                                      T* arr,
                                      const std::int64_t element_count,
                                      const T shift,
                                      const event_vector& deps = {}) {
    if (shift == T(0))
        return sycl::event();

    const size_t element_count_size_t = dal::detail::integral_cast<size_t>(element_count);
    const sycl::range<1> range{ element_count_size_t };

    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> id) {
            arr[id] += shift;
        });
    });
}

sycl::event shift_array_values(const detail::data_parallel_policy& policy,
                               void* arr,
                               data_type arr_type,
                               const std::int64_t element_count,
                               const void* shift,
                               const event_vector& deps) {
    ONEDAL_ASSERT(arr);
    ONEDAL_ASSERT(shift);
    ONEDAL_ASSERT(element_count > 0);

    return dispatch_by_data_type(arr_type, [&](auto arr_type_id) {
        using data_t = decltype(arr_type_id);
        data_t* data = static_cast<data_t*>(arr);
        const data_t shift_val = static_cast<const data_t*>(shift)[0];

        if (shift_val == data_t(0))
            return sycl::event();

        sycl::queue& q = policy.get_queue();
        if (is_device_friendly_usm(q, arr)) {
            return shift_array_values_device(q, data, element_count, shift_val, deps);
        }
        else {
            shift_array_values(data, element_count, shift_val);
            return sycl::event();
        }
    });
}

#endif

} // namespace oneapi::dal::backend
