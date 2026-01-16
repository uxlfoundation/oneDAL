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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type, typename Index>
sycl::event select_indexed(sycl::queue& q,
                           const ndview<Index, 2>& ids,
                           const ndview<Type, 2>& src,
                           ndview<Type, 2>& dst,
                           const event_vector& deps) {
    ONEDAL_ASSERT(ids.has_data());
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    ONEDAL_ASSERT(ids.get_shape() == dst.get_shape());
    ONEDAL_ASSERT(ids.get_dimension(0) == src.get_dimension(0));
    const ndshape<2> shape = ids.get_shape();
    const auto range = make_range_2d(shape[0], shape[1]);
    const auto* const ids_ptr = ids.get_data();
    const auto* const src_ptr = src.get_data();
    auto* const dst_ptr = dst.get_mutable_data();
    const auto ids_str = ids.get_leading_stride();
    const auto src_str = src.get_leading_stride();
    const auto dst_str = dst.get_leading_stride();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> idx) {
            const auto& index = *(ids_ptr + ids_str * idx[0] + idx[1]);
            const auto& value = *(src_ptr + src_str * idx[0] + index);
            *(dst_ptr + dst_str * idx[0] + idx[1]) = value;
        });
    });
}

template <typename Type, typename Index>
sycl::event select_indexed_naive(sycl::queue& q,
                                 const ndview<Index, 2>& ids,
                                 const ndview<Type, 1>& src,
                                 ndview<Type, 2>& dst,
                                 const event_vector& deps) {
    const ndshape<2> shape = ids.get_shape();
    const auto range = make_range_2d(shape[0], shape[1]);
    const auto* const ids_ptr = ids.get_data();
    const auto* const src_ptr = src.get_data();
    auto* const dst_ptr = dst.get_mutable_data();
    const auto ids_str = ids.get_leading_stride();
    const auto dst_str = dst.get_leading_stride();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> idx) {
            const auto& index = *(ids_ptr + ids_str * idx[0] + idx[1]);
            *(dst_ptr + dst_str * idx[0] + idx[1]) = *(src_ptr + index);
        });
    });
}

template <typename Type, typename Index>
sycl::event select_indexed_local(sycl::queue& q,
                                 const ndview<Index, 2>& ids,
                                 const ndview<Type, 1>& src,
                                 ndview<Type, 2>& dst,
                                 const event_vector& deps) {
    constexpr Type first_bit = Type(1) << (8 * sizeof(Type) - 1);
    const auto* const ids_ptr = ids.get_data();
    const auto* const src_ptr = src.get_data();
    auto* const dst_ptr = dst.get_mutable_data();
    const auto ids_str = ids.get_leading_stride();
    const auto dst_str = dst.get_leading_stride();
    const auto src_count = src.get_count();
    const auto row_count = ids.get_dimension(0);
    const auto col_count = ids.get_dimension(1);
    const auto wg_preffered = propose_wg_size(q);
    const auto width = std::min(wg_preffered, col_count);
    const auto wg_folding = wg_preffered / width + bool(wg_preffered & width);
    const auto block = device_local_mem_size(q) / (2 * sizeof(Type) * wg_folding);
    const std::int32_t block_count = src_count / block + bool(src_count % block);
    auto select_event = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        const auto nd_range = make_multiple_nd_range_2d({ block_count, width }, { 1l, width });
        sycl::local_accessor<Type, 1> cache{ make_range_1d(block), h };
        h.parallel_for(nd_range, [=](sycl::nd_item<2> it) {
            const auto bid = it.get_global_id(0);
            const std::int32_t from = bid * block;
            const std::int32_t to = (static_cast<std::int32_t>((bid + 1) * block) <
                                     static_cast<std::int32_t>(src_count))
                                        ? (bid + 1) * block
                                        : src_count;
            sycl::global_ptr<const Type> global((const Type*)(src_ptr + from));
            sycl::local_ptr<const Type> local(
                (const Type*)(cache.template get_multi_ptr<sycl::access::decorated::yes>()
                                  .get_raw()));
            it.async_work_group_copy(
                  cache.template get_multi_ptr<sycl::access::decorated::yes>(),
                  sycl::address_space_cast<sycl::access::address_space::global_space,
                                           sycl::access::decorated::yes>(src_ptr + from),
                  to - from)
                .wait();

            const auto cid = it.get_global_id(1);
            for (std::int32_t r = 0; r < row_count; ++r) {
                const auto idx = *(ids_ptr + ids_str * r + cid);
                const bool handle = (to > idx) && (idx >= from) && !bool(idx & first_bit);
                if (handle) {
                    *(dst_ptr + dst_str * r + cid) = local[idx - from] | first_bit;
                }
            }
        });
    });
    return q.submit([&](sycl::handler& h) {
        h.depends_on({ select_event });
        const auto range = make_range_2d(row_count, width);
        h.parallel_for(range, [=](sycl::id<2> idx) {
            *(dst_ptr + dst_str * idx[0] + idx[1]) &= (~first_bit);
        });
    });
}

template <typename Type, typename Index>
sycl::event select_indexed(sycl::queue& q,
                           const ndview<Index, 2>& ids,
                           const ndview<Type, 1>& src,
                           ndview<Type, 2>& dst,
                           const event_vector& deps) {
    ONEDAL_ASSERT(ids.has_data());
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    ONEDAL_ASSERT(ids.get_shape() == dst.get_shape());
    // TODO: investigate poorly performing select_indexed_local
    return select_indexed_naive(q, ids, src, dst, deps);
}

#define INSTANTIATE(TYPE, INDEX)                                 \
    template sycl::event select_indexed(sycl::queue&,            \
                                        const ndview<INDEX, 2>&, \
                                        const ndview<TYPE, 2>&,  \
                                        ndview<TYPE, 2>&,        \
                                        const event_vector&);    \
    template sycl::event select_indexed(sycl::queue&,            \
                                        const ndview<INDEX, 2>&, \
                                        const ndview<TYPE, 1>&,  \
                                        ndview<TYPE, 2>&,        \
                                        const event_vector&);

#define INSTANTIATE_TYPE(INDEX)       \
    INSTANTIATE(float, INDEX);        \
    INSTANTIATE(double, INDEX);       \
    INSTANTIATE(std::int32_t, INDEX); \
    INSTANTIATE(std::int64_t, INDEX);

INSTANTIATE_TYPE(std::int32_t);
INSTANTIATE_TYPE(std::int64_t);

} // namespace oneapi::dal::backend::primitives
