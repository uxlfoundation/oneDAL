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

#include "oneapi/dal/detail/threading.hpp"
#include "src/threading/threading.h"

ONEDAL_EXPORT int _onedal_threader_get_max_threads() {
    return _daal_threader_get_max_threads();
}

ONEDAL_EXPORT int _onedal_threader_get_current_thread_index() {
    return _daal_threader_get_current_thread_index();
}

ONEDAL_EXPORT void _onedal_threader_for(std::int64_t n,
                                        std::int64_t reserved,
                                        const void *a,
                                        oneapi::dal::preview::functype func) {
    _daal_threader_for(n, reserved, a, static_cast<daal::functype>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_simple(std::int64_t n,
                                               std::int64_t reserved,
                                               const void *a,
                                               oneapi::dal::preview::functype func) {
    _daal_threader_for_simple(n, reserved, a, static_cast<daal::functype>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_int64ptr(const std::int64_t *begin,
                                                 const std::int64_t *end,
                                                 const void *a,
                                                 oneapi::dal::preview::functype_int64ptr func) {
    _daal_threader_for_int64ptr(begin, end, a, static_cast<daal::functype_int64ptr>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_blocked(std::int64_t count,
                                                std::int64_t block,
                                                const void *a,
                                                oneapi::dal::preview::functype2 func) {
    _daal_threader_for_blocked(count, block, a, static_cast<daal::functype2>(func));
}

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32_int64(
    std::int32_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func) {
    return _daal_parallel_reduce_int32_int64(
        n,
        (std::int64_t)init,
        a,
        static_cast<daal::loop_functype_int32_int64>(loop_func),
        b,
        static_cast<daal::reduction_functype_int64>(reduction_func));
}

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32_int64_simple(
    std::int32_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func) {
    return _daal_parallel_reduce_int32_int64_simple(
        n,
        (std::int64_t)init,
        a,
        static_cast<daal::loop_functype_int32_int64>(loop_func),
        b,
        static_cast<daal::reduction_functype_int64>(reduction_func));
}

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32ptr_int64_simple(
    const std::int32_t *begin,
    const std::int32_t *end,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32ptr_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func) {
    return _daal_parallel_reduce_int32ptr_int64_simple(
        begin,
        end,
        (std::int64_t)init,
        a,
        static_cast<daal::loop_functype_int32ptr_int64>(loop_func),
        b,
        static_cast<daal::reduction_functype_int64>(reduction_func));
}

ONEDAL_EXPORT void *_onedal_get_tls_ptr(void *a, oneapi::dal::preview::tls_functype func) {
    return _daal_get_tls_ptr(a, func);
}
ONEDAL_EXPORT void *_onedal_get_tls_local(void *tlsPtr) {
    return _daal_get_tls_local(tlsPtr);
}
ONEDAL_EXPORT
void _onedal_reduce_tls(void *tlsPtr, void *a, oneapi::dal::preview::tls_reduce_functype func) {
    _daal_reduce_tls(tlsPtr, a, func);
}
ONEDAL_EXPORT void _onedal_parallel_reduce_tls(void *tlsPtr,
                                               void *a,
                                               oneapi::dal::preview::tls_reduce_functype func) {
    _daal_parallel_reduce_tls(tlsPtr, a, func);
}
ONEDAL_EXPORT void _onedal_del_tls_ptr(void *tlsPtr) {
    _daal_del_tls_ptr(tlsPtr);
}

ONEDAL_EXPORT void *_onedal_new_mutex() {
    return _daal_new_mutex();
}

ONEDAL_EXPORT void _onedal_lock_mutex(void *mutex_ptr) {
    _daal_lock_mutex(mutex_ptr);
}

ONEDAL_EXPORT void _onedal_unlock_mutex(void *mutex_ptr) {
    _daal_unlock_mutex(mutex_ptr);
}

ONEDAL_EXPORT void _onedal_del_mutex(void *mutex_ptr) {
    _daal_del_mutex(mutex_ptr);
}

namespace oneapi::dal::detail {

typedef std::pair<std::int32_t, std::size_t> pair_int32_t_size_t;

#define ONEDAL_PARALLEL_SORT_SPECIALIZATION(TYPE, DAALTYPE, NAMESUFFIX)               \
    template <>                                                                       \
    ONEDAL_EXPORT void parallel_sort(TYPE *begin_ptr, TYPE *end_ptr) {                \
        static_assert(sizeof(TYPE) == sizeof(DAALTYPE));                              \
        _daal_parallel_sort_##NAMESUFFIX((DAALTYPE *)begin_ptr, (DAALTYPE *)end_ptr); \
    }

ONEDAL_PARALLEL_SORT_SPECIALIZATION(std::int32_t, int, int32)
ONEDAL_PARALLEL_SORT_SPECIALIZATION(std::uint64_t, std::size_t, uint64)
ONEDAL_PARALLEL_SORT_SPECIALIZATION(pair_int32_t_size_t, daal::IdxValType<int>, pair_int32_uint64)

#undef ONEDAL_PARALLEL_SORT_SPECIALIZATION

} // namespace oneapi::dal::detail
