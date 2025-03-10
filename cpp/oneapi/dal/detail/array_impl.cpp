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

#include "oneapi/dal/detail/array_impl.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::detail::v2 {

using deserialize_result_t = std::tuple<shared<byte_t>, std::int64_t>;

inline void serialize_array_on_host(output_archive& archive,
                                    const byte_t* data,
                                    std::int64_t size_in_bytes,
                                    data_type dtype) {
    archive(ONEDAL_SERIALIZATION_ID(array_id));
    archive(dtype, size_in_bytes);

    if (size_in_bytes > 0) {
        ONEDAL_ASSERT(data);
        archive.range(data, data + size_in_bytes);
    }
}

inline deserialize_result_t deserialize_array_on_host(input_archive& archive,
                                                      data_type expected_dtype) {
    const std::uint64_t serialization_id = archive.pop<std::uint64_t>();
    if (serialization_id != ONEDAL_SERIALIZATION_ID(array_id)) {
        throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
    }

    const data_type dtype = archive.pop<data_type>();
    if (dtype != expected_dtype) {
        throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
    }

    const std::int64_t size_in_bytes = archive.pop<std::int64_t>();
    if (size_in_bytes < 0 || size_in_bytes % get_data_type_size(expected_dtype) > 0) {
        throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
    }

    if (size_in_bytes > 0) {
        auto deleter = make_default_delete<byte_t>(detail::default_host_policy{});
        byte_t* data_placeholder = malloc<byte_t>(detail::default_host_policy{}, size_in_bytes);
        auto shared_data_placeholder = shared<byte_t>{ data_placeholder, std::move(deleter) };

        archive.range(data_placeholder, data_placeholder + size_in_bytes);

        return { shared_data_placeholder, size_in_bytes };
    }
    else {
        return { shared<byte_t>{}, size_in_bytes };
    }
}

inline void serialize_array_impl(const default_host_policy& policy,
                                 output_archive& archive,
                                 const byte_t* data,
                                 std::int64_t size_in_bytes,
                                 data_type dtype) {
    serialize_array_on_host(archive, data, size_in_bytes, dtype);
}

inline deserialize_result_t deserialize_array_impl(const default_host_policy& policy,
                                                   input_archive& archive,
                                                   data_type expected_dtype) {
    return deserialize_array_on_host(archive, expected_dtype);
}

#ifdef ONEDAL_DATA_PARALLEL
inline void serialize_array_impl(const data_parallel_policy& policy,
                                 output_archive& archive,
                                 const byte_t* data,
                                 std::int64_t size_in_bytes,
                                 data_type dtype) {
    auto& q = policy.get_queue();

    if (backend::is_device_usm(q, data)) {
        const auto host_buffer = backend::make_unique_usm_host<byte_t>(q, size_in_bytes);
        backend::copy(q, host_buffer.get(), data, size_in_bytes).wait_and_throw();
        serialize_array_on_host(archive, host_buffer.get(), size_in_bytes, dtype);
    }
    else {
        serialize_array_on_host(archive, data, size_in_bytes, dtype);
    }
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
inline deserialize_result_t deserialize_array_impl(const data_parallel_policy& policy,
                                                   input_archive& archive,
                                                   data_type expected_dtype) {
    auto& q = policy.get_queue();

    const auto [shared_data_host, size_in_bytes] =
        deserialize_array_on_host(archive, expected_dtype);

    if (size_in_bytes > 0) {
        auto deleter = make_default_delete<byte_t>(policy);
        byte_t* data_device = backend::malloc_device<byte_t>(q, size_in_bytes);
        auto shared_data_device = shared<byte_t>{ data_device, std::move(deleter) };

        backend::copy_host2usm(q, data_device, shared_data_host.get(), size_in_bytes);

        return { shared_data_device, size_in_bytes };
    }
    else {
        ONEDAL_ASSERT(shared_data_host.get() == nullptr);
        return { shared_data_host, size_in_bytes };
    }
}
#endif

template <typename Policy>
void serialize_array(const Policy& policy,
                     output_archive& archive,
                     const byte_t* data,
                     std::int64_t size_in_bytes,
                     data_type dtype) {
    serialize_array_impl(policy, archive, data, size_in_bytes, dtype);
}

template <typename Policy>
deserialize_result_t deserialize_array(const Policy& policy,
                                       input_archive& archive,
                                       data_type expected_dtype) {
    return deserialize_array_impl(policy, archive, expected_dtype);
}

#define INSTANTIATE(Policy)                                                               \
    template void ONEDAL_EXPORT serialize_array(const Policy& policy,                     \
                                                output_archive& archive,                  \
                                                const byte_t* data,                       \
                                                std::int64_t size_in_bytes,               \
                                                data_type dtype);                         \
    template deserialize_result_t ONEDAL_EXPORT deserialize_array(const Policy& policy,   \
                                                                  input_archive& archive, \
                                                                  data_type expected_dtype);

INSTANTIATE(default_host_policy)

#ifdef ONEDAL_DATA_PARALLEL
INSTANTIATE(data_parallel_policy)
#endif

} // namespace oneapi::dal::detail::v2
