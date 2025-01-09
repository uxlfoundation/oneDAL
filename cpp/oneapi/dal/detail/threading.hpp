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

#include <utility>
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/exceptions.hpp"

#if defined(_WIN32) || defined(_WIN64)
#include <intrin.h>
#endif

namespace oneapi::dal::preview {
typedef void (*functype)(std::int32_t i, const void *a);
typedef void (*functype_int64)(std::int64_t i, const void *a);
typedef void (*functype_int32ptr)(const std::int32_t *i, const void *a);
typedef void (*functype_blocked_size)(std::size_t, std::size_t, const void *);
typedef void *(*tls_functype)(const void *a);
typedef void (*tls_reduce_functype)(void *p, const void *a);

typedef std::int64_t (*loop_functype_int32_int64)(std::int32_t start_idx,
                                                  std::int32_t end_idx,
                                                  std::int64_t value_for_reduce,
                                                  const void *a);

typedef std::int64_t (*loop_functype_int32ptr_int64)(const std::int32_t *begin,
                                                     const std::int32_t *end,
                                                     std::int64_t value_for_reduce,
                                                     const void *a);

typedef std::int64_t (*reduction_functype_int64)(std::int64_t a,
                                                 std::int64_t b,
                                                 const void *reduction);

typedef std::pair<std::int32_t, std::size_t> pair_int32_t_size_t;
} // namespace oneapi::dal::preview

extern "C" {
ONEDAL_EXPORT int _onedal_threader_get_max_threads();

ONEDAL_EXPORT int _onedal_threader_get_current_thread_index();

ONEDAL_EXPORT void _onedal_threader_for(std::int32_t n,
                                        std::int32_t threads_request,
                                        const void *a,
                                        oneapi::dal::preview::functype func);

ONEDAL_EXPORT void _onedal_threader_for_int64(std::int64_t n,
                                              const void *a,
                                              oneapi::dal::preview::functype_int64 func);

ONEDAL_EXPORT void _onedal_threader_for_simple(std::int32_t n,
                                               std::int32_t threads_request,
                                               const void *a,
                                               oneapi::dal::preview::functype func);

ONEDAL_EXPORT void _onedal_threader_for_int32ptr(const std::int32_t *begin,
                                                 const std::int32_t *end,
                                                 const void *a,
                                                 oneapi::dal::preview::functype_int32ptr func);

ONEDAL_EXPORT void _onedal_threader_for_blocked_size(
    std::size_t count,
    std::size_t block,
    const void *a,
    oneapi::dal::preview::functype_blocked_size func);

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32_int64(
    std::int32_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func);

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32_int64_simple(
    std::int32_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func);

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32ptr_int64_simple(
    const std::int32_t *begin,
    const std::int32_t *end,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32ptr_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func);

ONEDAL_EXPORT void *_onedal_get_tls_ptr(void *a, oneapi::dal::preview::tls_functype func);
ONEDAL_EXPORT void *_onedal_get_tls_local(void *tlsPtr);
ONEDAL_EXPORT void _onedal_reduce_tls(void *tlsPtr,
                                      void *a,
                                      oneapi::dal::preview::tls_reduce_functype func);
ONEDAL_EXPORT void _onedal_parallel_reduce_tls(void *tlsPtr,
                                               void *a,
                                               oneapi::dal::preview::tls_reduce_functype func);
ONEDAL_EXPORT void _onedal_del_tls_ptr(void *tlsPtr);

ONEDAL_EXPORT void *_onedal_new_mutex();
ONEDAL_EXPORT void _onedal_lock_mutex(void *mutex_ptr);
ONEDAL_EXPORT void _onedal_unlock_mutex(void *mutex_ptr);
ONEDAL_EXPORT void _onedal_del_mutex(void *mutex_ptr);
}

namespace oneapi::dal::detail {
inline int threader_get_max_threads() {
    return _onedal_threader_get_max_threads();
}

inline int threader_get_current_thread_index() {
    return _onedal_threader_get_current_thread_index();
}

template <typename F>
inline void threader_func(std::int32_t i, const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    lambda(i);
}

template <typename F>
inline void threader_func_int64(std::int64_t i, const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    lambda(i);
}

template <typename F>
inline void threader_func_int32ptr(const std::int32_t *i, const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    lambda(i);
}

template <typename F>
inline void threader_func_blocked_size(std::size_t f, std::size_t l, const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    lambda(f, l);
}

template <typename F>
inline ONEDAL_EXPORT void threader_for(std::int32_t n,
                                       std::int32_t threads_request,
                                       const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _onedal_threader_for(n, threads_request, a, threader_func<F>);
}

template <typename F>
inline ONEDAL_EXPORT void threader_for_int64(std::int64_t n, const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _onedal_threader_for_int64(n, a, threader_func_int64<F>);
}

template <typename F>
inline ONEDAL_EXPORT void threader_for_simple(std::int32_t n,
                                              std::int32_t threads_request,
                                              const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _onedal_threader_for_simple(n, threads_request, a, threader_func<F>);
}

template <typename F>
inline ONEDAL_EXPORT void threader_for_int32ptr(const std::int32_t *begin,
                                                const std::int32_t *end,
                                                const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _onedal_threader_for_int32ptr(begin, end, a, threader_func_int32ptr<F>);
}

template <typename F>
inline ONEDAL_EXPORT void threader_for_blocked_size(std::size_t count,
                                                    std::size_t block,
                                                    const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _onedal_threader_for_blocked_size(count, block, a, threader_func_blocked_size<F>);
}

template <typename F>
inline std::int64_t parallel_reduce_loop_int32_int64(std::int32_t start_idx,
                                                     std::int32_t end_idx,
                                                     std::int64_t value_for_reduce,
                                                     const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    return lambda(start_idx, end_idx, value_for_reduce);
}

template <typename F>
inline std::int64_t parallel_reduce_loop_int32ptr_int64(const std::int32_t *start_idx,
                                                        const std::int32_t *end_idx,
                                                        std::int64_t value_for_reduce,
                                                        const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    return lambda(start_idx, end_idx, value_for_reduce);
}

template <typename F>
inline std::int64_t parallel_reduce_reduction_int64(std::int64_t a,
                                                    std::int64_t b,
                                                    const void *reduction) {
    const F &lambda = *static_cast<const F *>(reduction);
    return lambda(a, b);
}

template <typename Value, typename Func, typename Reduction>
inline Value parallel_reduce_int32_int64_t(std::int32_t n,
                                           Value init,
                                           const Func &func,
                                           const Reduction &reduction) {
    const void *const lf = static_cast<const void *>(&func);
    const void *const rf = static_cast<const void *>(&reduction);

    return _onedal_parallel_reduce_int32_int64(n,
                                               init,
                                               lf,
                                               parallel_reduce_loop_int32_int64<Func>,
                                               rf,
                                               parallel_reduce_reduction_int64<Reduction>);
}

template <typename Value, typename Func, typename Reduction>
inline Value parallel_reduce_int32_int64_t_simple(std::int32_t n,
                                                  Value init,
                                                  const Func &func,
                                                  const Reduction &reduction) {
    const void *const lf = static_cast<const void *>(&func);
    const void *const rf = static_cast<const void *>(&reduction);

    return _onedal_parallel_reduce_int32_int64_simple(n,
                                                      init,
                                                      lf,
                                                      parallel_reduce_loop_int32_int64<Func>,
                                                      rf,
                                                      parallel_reduce_reduction_int64<Reduction>);
}

template <typename Value, typename Func, typename Reduction>
inline Value parallel_reduce_int32ptr_int64_t_simple(const std::int32_t *begin,
                                                     const std::int32_t *end,
                                                     Value init,
                                                     const Func &func,
                                                     const Reduction &reduction) {
    const void *const lf = static_cast<const void *>(&func);
    const void *const rf = static_cast<const void *>(&reduction);

    return _onedal_parallel_reduce_int32ptr_int64_simple(
        begin,
        end,
        init,
        lf,
        parallel_reduce_loop_int32ptr_int64<Func>,
        rf,
        parallel_reduce_reduction_int64<Reduction>);
}

template <typename F>
ONEDAL_EXPORT void parallel_sort(F *begin_ptr, F *end_ptr) {
    throw unimplemented(dal::detail::error_messages::unimplemented_sorting_procedure());
}

#define ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL(TYPE) \
    template <>                                        \
    ONEDAL_EXPORT void parallel_sort(TYPE *begin_ptr, TYPE *end_ptr);

ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL(std::int32_t)
ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL(std::uint64_t)
ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL(oneapi::dal::preview::pair_int32_t_size_t)

#undef ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL

inline void atomic_increment(std::int64_t &value, std::int64_t delta = 1) {
#if defined(_WIN32) || defined(_WIN64)
    _InterlockedExchangeAdd64(&value, delta);
#else
    __atomic_add_fetch(&value, delta, __ATOMIC_SEQ_CST);
#endif
}

inline void atomic_decrement(std::int64_t &value, std::int64_t delta = 1) {
#if defined(_WIN32) || defined(_WIN64)
    _InterlockedExchangeAdd64(&value, -delta);
#else
    __atomic_sub_fetch(&value, delta, __ATOMIC_SEQ_CST);
#endif
}

inline std::int64_t atomic_load(std::int64_t &value) {
#if defined(_WIN32) || defined(_WIN64)
    const std::int64_t result = value;
    _ReadWriteBarrier();
    return result;
#else
    const std::int64_t result = __atomic_load_n(&value, __ATOMIC_ACQUIRE);
    __asm__ __volatile__("" : : : "memory");
    return result;
#endif
}

template <typename lambdaType>
inline void *tls_func(const void *a) {
    const lambdaType &lambda = *static_cast<const lambdaType *>(a);
    return lambda();
}

template <typename F, typename lambdaType>
inline void tls_reduce_func(void *v, const void *a) {
    const lambdaType &lambda = *static_cast<const lambdaType *>(a);
    lambda((F)v);
}

struct tlsBase {
    virtual ~tlsBase() {}
};

class tls_deleter : public tlsBase {
public:
    virtual ~tls_deleter() {}
    virtual void del(void *a) = 0;
};

template <typename lambdaType>
class tls_deleter_ : public tls_deleter {
public:
    virtual ~tls_deleter_() {}
    virtual void del(void *a) {
        delete static_cast<lambdaType *>(a);
    }
};

template <typename F>
class tls : public tlsBase {
public:
    template <typename lambdaType>
    explicit tls(const lambdaType &lambda) {
        lambdaType *locall = new lambdaType(lambda);
        d = new tls_deleter_<lambdaType>();

        const void *ac = static_cast<const void *>(locall);
        void *a = const_cast<void *>(ac);
        voidLambda = a;

        tlsPtr = _onedal_get_tls_ptr(a, tls_func<lambdaType>);
    }

    virtual ~tls() {
        d->del(voidLambda);
        delete d;
        _onedal_del_tls_ptr(tlsPtr);
    }

    F local() {
        void *pf = _onedal_get_tls_local(tlsPtr);
        return (static_cast<F>(pf));
    }

    template <typename lambdaType>
    void reduce(const lambdaType &lambda) {
        const void *ac = static_cast<const void *>(&lambda);
        void *a = const_cast<void *>(ac);
        _onedal_reduce_tls(tlsPtr, a, tls_reduce_func<F, lambdaType>);
    }

    template <typename lambdaType>
    void parallel_reduce(const lambdaType &lambda) {
        const void *ac = static_cast<const void *>(&lambda);
        void *a = const_cast<void *>(ac);
        _onedal_parallel_reduce_tls(tlsPtr, a, tls_reduce_func<F, lambdaType>);
    }

private:
    void *tlsPtr;
    void *voidLambda;
    tls_deleter *d;
};

template <typename T, typename Allocator = std::allocator<char>>
class tls_mem : public oneapi::dal::detail::tls<T *> {
public:
    typedef oneapi::dal::detail::tls<T *> super;
    // tls_mem(Allocator allocator, std::size_t count = 1)
    tls_mem(std::size_t count = 1)
            : super([=]() -> T * {
                  using t_allocator_type =
                      typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
                  std::allocator<char> local_char_alloc; //WRONG
                  t_allocator_type local_allocator(local_char_alloc);
                  using allocator_traits_t =
                      typename std::allocator_traits<Allocator>::template rebind_traits<T>;
                  typename allocator_traits_t::pointer ptr =
                      allocator_traits_t::allocate(local_allocator, count);
                  if (ptr == nullptr) {
                      throw host_bad_alloc();
                  }
                  return (T *)ptr;
              }),
              _count(count) {
        Allocator allocator;
        alloc_ = allocator;
    }

    ~tls_mem() {
        super::reduce([&](T *ptr) -> void {
            using t_allocator_type =
                typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
            t_allocator_type t_allocator(alloc_);
            using allocator_traits_t =
                typename std::allocator_traits<Allocator>::template rebind_traits<T>;
            if (ptr != nullptr) {
                allocator_traits_t::deallocate(t_allocator, ptr, _count);
            }
        });
    }

    using ptr_t = typename std::add_pointer_t<T>;
    using reference = ptr_t;

private:
    Allocator alloc_;
    std::size_t _count;
};

class mutex {
public:
    mutex() : impl_(_onedal_new_mutex()) {}
    mutex(const mutex &) = delete;
    mutex &operator=(const mutex &) = delete;
    ~mutex() {
        if (impl_) {
            _onedal_del_mutex(impl_);
        }
    }
    void lock() {
        if (impl_) {
            _onedal_lock_mutex(impl_);
        }
    }
    void unlock() {
        if (impl_) {
            _onedal_unlock_mutex(impl_);
        }
    }

private:
    void *impl_;
};

class scoped_lock {
public:
    explicit scoped_lock(mutex &m) : mutex_(m) {
        mutex_.lock();
    }
    scoped_lock(const scoped_lock &) = delete;
    scoped_lock(scoped_lock &&) = delete;
    scoped_lock &operator=(scoped_lock &other) = delete;
    scoped_lock &operator=(scoped_lock &&other) = delete;
    ~scoped_lock() {
        mutex_.unlock();
    }

private:
    mutex &mutex_;
};

} // namespace oneapi::dal::detail
