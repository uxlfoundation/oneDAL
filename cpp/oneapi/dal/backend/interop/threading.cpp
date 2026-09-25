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

/* --------------------------------------------------------------------------------------------
 * Frozen 32-bit entry points.
 *
 * Nothing in the tree calls these -- every call site goes through the 64-bit loops below -- but
 * they are exported from `libonedal` and reachable from an application's own object files (see
 * the comment above their declarations in `oneapi/dal/detail/threading.hpp`), so their symbols
 * and signatures are part of the ABI. Each one reproduces the behaviour it had before the
 * threading layer moved to 64-bit indices: the second argument was documented as reserved and
 * ignored by the backend, which hardcoded a grain size of 1.
 * ----------------------------------------------------------------------------------------- */

ONEDAL_EXPORT void _onedal_threader_for(std::int32_t n,
                                        std::int32_t /* reserved */,
                                        const void *a,
                                        oneapi::dal::preview::functype func) {
    _daal_threader_for_int32(n, 1, a, static_cast<daal::functype_int32>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_int64(std::int64_t n,
                                              const void *a,
                                              oneapi::dal::preview::functype_int64 func) {
    _daal_threader_for(n, 1, a, static_cast<daal::functype>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_simple(std::int32_t n,
                                               std::int32_t /* reserved */,
                                               const void *a,
                                               oneapi::dal::preview::functype func) {
    _daal_threader_for_simple_int32(n, 1, a, static_cast<daal::functype_int32>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_int32ptr(const std::int32_t *begin,
                                                 const std::int32_t *end,
                                                 const void *a,
                                                 oneapi::dal::preview::functype_int32ptr func) {
    // The 32-bit pointer-range loop in the DAAL threading layer is gone, so this iterates over
    // the range by index instead. `func` still receives a pointer into `[begin, end)`, one
    // element at a time and with a grain size of 1, which is what the removed
    // `_daal_threader_for_int32ptr` did.
    const std::int64_t count = static_cast<std::int64_t>(end - begin);
    daal::threader_for(count, 1, [&](std::int64_t i) -> void {
        func(begin + i, a);
    });
}

ONEDAL_EXPORT void _onedal_threader_for_blocked_size(
    std::size_t count,
    std::size_t block,
    const void *a,
    oneapi::dal::preview::functype_blocked_size func) {
    // `_daal_threader_for_blocked` replaced `_daal_threader_for_blocked_size` and carries the
    // same `(first, last)` callback convention, so only the index type has to be adapted.
    daal::threader_for_blocked(
        static_cast<std::int64_t>(count),
        static_cast<std::int64_t>(block),
        [&](std::int64_t first, std::int64_t last) -> void {
            func(static_cast<std::size_t>(first), static_cast<std::size_t>(last), a);
        });
}

/* ------------------------------- 64-bit loops (the default) ------------------------------- */

ONEDAL_EXPORT void _onedal_threader_for_with_grain(std::int64_t n,
                                                   std::int64_t grain_size,
                                                   const void *a,
                                                   oneapi::dal::preview::functype_int64 func) {
    _daal_threader_for(n, grain_size, a, static_cast<daal::functype>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_simple_with_grain(
    std::int64_t n,
    std::int64_t grain_size,
    const void *a,
    oneapi::dal::preview::functype_int64 func) {
    _daal_threader_for_simple(n, grain_size, a, static_cast<daal::functype>(func));
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
                                                oneapi::dal::preview::functype_blocked func) {
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
