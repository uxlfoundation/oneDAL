/* file: core_threading_win_dll.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of "stubs" for threading layer functions for win dll case.
//--
*/

#include "services/internal/daal_load_win_dynamic_lib.h"
#include "src/threading/threading.h"
#include "src/threading/service_thread_pinner.h"
#include "services/env_detect.h"

static HMODULE daal_thr_dll_handle = NULL;
daal::services::Environment::LibraryThreadingType __daal_serv_get_thr_set();

#define DAAL_LOAD_DLL(name) _daal_load_win_dynamic_lib(name)

DAAL_EXPORT HMODULE load_onedal_thread_dll()
{
#ifdef _DEBUG
    return DAAL_LOAD_DLL("onedal_threadd.3.dll");
#else
    return DAAL_LOAD_DLL("onedal_thread.3.dll");
#endif
}

static void load_daal_thr_dll(void)
{
    if (daal_thr_dll_handle != NULL)
    {
        return;
    }

    switch (__daal_serv_get_thr_set())
    {
    case daal::services::Environment::MultiThreaded:
    {
        daal_thr_dll_handle = load_onedal_thread_dll();
        if (daal_thr_dll_handle == NULL)
        {
            printf("Intel oneDAL FATAL ERROR: Cannot load onedal_thread.3.dll.\n");
            exit(1);
        }
        break;
    }
    default:
    {
        daal_thr_dll_handle = load_onedal_thread_dll();
        if (daal_thr_dll_handle != NULL)
        {
            return;
        }

        printf("Intel oneDAL FATAL ERROR: Cannot load onedal_thread.3.dll.\n");
        exit(1);
    }
    }
}

FARPROC load_daal_thr_func(const char * ordinal)
{
    FARPROC FuncAddress;

    if (daal_thr_dll_handle == NULL)
    {
        printf("Intel oneDAL FATAL ERROR: Cannot load \"%s\" function because threaded layer DLL isn`t loaded.\n", ordinal);
        exit(1);
    }

    FuncAddress = GetProcAddress(daal_thr_dll_handle, ordinal);
    if (FuncAddress == NULL)
    {
        printf("GetLastError error code is %lx\n", GetLastError());
        printf("Intel oneDAL FATAL ERROR: Cannot load \"%s\" function.\n", ordinal);
        exit(1);
    }

    return FuncAddress;
}

typedef void * (*_threaded_malloc_t)(const size_t, const size_t);
typedef void (*_threaded_free_t)(void *);

typedef void (*_daal_threader_for_t)(int, int, const void *, daal::functype);
typedef void (*_daal_threader_for_int64_t)(int64_t, const void *, daal::functype_int64);
typedef void (*_daal_threader_for_int32ptr_t)(const int *, const int *, const void *, daal::functype_int32ptr);
typedef void (*_daal_threader_for_simple_t)(int, int, const void *, daal::functype);
typedef void (*_daal_static_threader_for_t)(size_t, const void *, daal::functype_static);
typedef void (*_daal_threader_for_blocked_t)(int, int, const void *, daal::functype2);
typedef void (*_daal_threader_for_blocked_size_t)(size_t, size_t, const void *, daal::functype_blocked_size);
typedef int (*_daal_threader_get_max_threads_t)(void);
typedef int (*_daal_threader_get_current_thread_index_t)(void);
typedef void (*_daal_threader_for_break_t)(int, int, const void *, daal::functype_break);

typedef int64_t (*_daal_parallel_reduce_int32_int64_t)(int32_t, int64_t, const void *, daal::loop_functype_int32_int64, const void *,
                                                       daal::reduction_functype_int64);
typedef int64_t (*_daal_parallel_reduce_int32_int64_t_simple)(int32_t, int64_t, const void *, daal::loop_functype_int32_int64, const void *,
                                                              daal::reduction_functype_int64);
typedef int64_t (*_daal_parallel_reduce_int32ptr_int64_t_simple)(const int32_t *, const int32_t *, int64_t, const void *,
                                                                 daal::loop_functype_int32ptr_int64, const void *, daal::reduction_functype_int64);

typedef void * (*_daal_get_tls_ptr_t)(void *, daal::tls_functype);
typedef void (*_daal_del_tls_ptr_t)(void *);
typedef void * (*_daal_get_tls_local_t)(void *);
typedef void (*_daal_reduce_tls_t)(void *, void *, daal::tls_reduce_functype);
typedef void (*_daal_parallel_reduce_tls_t)(void *, void *, daal::tls_reduce_functype);

typedef void * (*_daal_get_ls_ptr_t)(void *, daal::tls_functype);
typedef void (*_daal_del_ls_ptr_t)(void *);
typedef void * (*_daal_get_ls_local_t)(void *);
typedef void (*_daal_release_ls_local_t)(void *, void *);
typedef void (*_daal_reduce_ls_t)(void *, void *, daal::tls_reduce_functype);

typedef void * (*_daal_new_mutex_t)();
typedef void (*_daal_del_mutex_t)(void *);
typedef void (*_daal_lock_mutex_t)(void *);
typedef void (*_daal_unlock_mutex_t)(void *);

typedef void * (*_daal_new_task_group_t)();
typedef void (*_daal_del_task_group_t)(void * taskGroupPtr);
typedef void (*_daal_run_task_group_t)(void * taskGroupPtr, daal::task * t);
typedef void (*_daal_wait_task_group_t)(void * taskGroupPtr);

typedef bool (*_daal_is_in_parallel_t)();
typedef void (*_daal_tbb_task_scheduler_free_t)(void *& globalControl);
typedef void (*_daal_tbb_task_scheduler_handle_free_t)(void *& schedulerHandle);
typedef size_t (*_setNumberOfThreads_t)(const size_t, void **);
typedef size_t (*_setSchedulerHandle_t)(void **);
typedef void * (*_daal_threader_env_t)();

typedef void (*_daal_parallel_sort_int32_t)(int *, int *);
typedef void (*_daal_parallel_sort_uint64_t)(size_t *, size_t *);
typedef void (*_daal_parallel_sort_pair_int32_uint64_t)(daal::IdxValType<int> *, daal::IdxValType<int> *);
typedef void (*_daal_parallel_sort_pair_fp32_uint64_t)(daal::IdxValType<float> *, daal::IdxValType<float> *);
typedef void (*_daal_parallel_sort_pair_fp64_uint64_t)(daal::IdxValType<double> *, daal::IdxValType<double> *);

#if !(defined DAAL_THREAD_PINNING_DISABLED)
typedef void (*_thread_pinner_thread_pinner_init_t)();
typedef void (*_thread_pinner_read_topology_t)();
typedef void (*_thread_pinner_on_scheduler_entry_t)(bool);
typedef void (*_thread_pinner_on_scheduler_exit_t)(bool);
typedef void (*_thread_pinner_execute_t)(daal::services::internal::thread_pinner_task_t & f);
typedef int (*_thread_pinner_get_status_t)();
typedef bool (*_thread_pinner_get_pinning_t)();
typedef bool (*_thread_pinner_set_pinning_t)(bool p);
typedef void * (*_getThreadPinner_t)(bool create_pinner, void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *));
#endif

typedef void (*_daal_threader_reduce_t)(const size_t, const size_t, daal::Reducer & reducer);
typedef void (*_daal_static_threader_reduce_t)(const size_t, const size_t, daal::Reducer & reducer);

DAAL_EXPORT void * _threaded_scalable_malloc(const size_t size, const size_t alignment)
{
    load_daal_thr_dll();
    static _threaded_malloc_t _threaded_malloc_ptr = (_threaded_malloc_t)load_daal_thr_func("_threaded_scalable_malloc");
    return _threaded_malloc_ptr(size, alignment);
}

DAAL_EXPORT void _threaded_scalable_free(void * ptr)
{
    load_daal_thr_dll();
    static _threaded_free_t _threaded_free_ptr = (_threaded_free_t)load_daal_thr_func("_threaded_scalable_free");
    _threaded_free_ptr(ptr);
}

DAAL_EXPORT void _daal_threader_for(int n, int threads_request, const void * a, daal::functype func)
{
    load_daal_thr_dll();
    static _daal_threader_for_t _daal_threader_for_ptr = (_daal_threader_for_t)load_daal_thr_func("_daal_threader_for");
    _daal_threader_for_ptr(n, threads_request, a, func);
}

DAAL_EXPORT void _daal_threader_for_simple(int n, int threads_request, const void * a, daal::functype func)
{
    load_daal_thr_dll();
    static _daal_threader_for_simple_t _daal_threader_for_simple_ptr = (_daal_threader_for_simple_t)load_daal_thr_func("_daal_threader_for_simple");
    _daal_threader_for_simple_ptr(n, threads_request, a, func);
}

DAAL_EXPORT void _daal_threader_for_int32ptr(const int * begin, const int * end, const void * a, daal::functype_int32ptr func)
{
    load_daal_thr_dll();

    static _daal_threader_for_int32ptr_t _daal_threader_for_int32ptr_ptr =
        (_daal_threader_for_int32ptr_t)load_daal_thr_func("_daal_threader_for_int32ptr");
    _daal_threader_for_int32ptr_ptr(begin, end, a, func);
}

DAAL_EXPORT void _daal_threader_for_int64(int64_t n, const void * a, daal::functype_int64 func)
{
    load_daal_thr_dll();
    static _daal_threader_for_int64_t _daal_threader_for_int64_ptr = (_daal_threader_for_int64_t)load_daal_thr_func("_daal_threader_for_int64");
    _daal_threader_for_int64_ptr(n, a, func);
}

DAAL_EXPORT void _daal_static_threader_for(size_t n, const void * a, daal::functype_static func)
{
    load_daal_thr_dll();
    static _daal_static_threader_for_t _daal_static_threader_for_ptr = (_daal_static_threader_for_t)load_daal_thr_func("_daal_static_threader_for");
    _daal_static_threader_for_ptr(n, a, func);
}

DAAL_EXPORT void _daal_parallel_sort_int32(int * begin_ptr, int * end_ptr)
{
    load_daal_thr_dll();
    static _daal_parallel_sort_int32_t _daal_parallel_sort_int32_ptr = (_daal_parallel_sort_int32_t)load_daal_thr_func("_daal_parallel_sort_int32");
    _daal_parallel_sort_int32_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_parallel_sort_uint64(size_t * begin_ptr, size_t * end_ptr)
{
    load_daal_thr_dll();
    static _daal_parallel_sort_uint64_t _daal_parallel_sort_uint64_ptr =
        (_daal_parallel_sort_uint64_t)load_daal_thr_func("_daal_parallel_sort_uint64");
    _daal_parallel_sort_uint64_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_parallel_sort_pair_int32_uint64(daal::IdxValType<int> * begin_ptr, daal::IdxValType<int> * end_ptr)
{
    load_daal_thr_dll();
    static _daal_parallel_sort_pair_int32_uint64_t _daal_parallel_sort_pair_int32_uint64_ptr =
        (_daal_parallel_sort_pair_int32_uint64_t)load_daal_thr_func("_daal_parallel_sort_pair_int32_uint64");
    _daal_parallel_sort_pair_int32_uint64_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_parallel_sort_pair_fp32_uint64(daal::IdxValType<float> * begin_ptr, daal::IdxValType<float> * end_ptr)
{
    load_daal_thr_dll();
    static _daal_parallel_sort_pair_fp32_uint64_t _daal_parallel_sort_pair_fp32_uint64_ptr =
        (_daal_parallel_sort_pair_fp32_uint64_t)load_daal_thr_func("_daal_parallel_sort_pair_fp32_uint64");
    _daal_parallel_sort_pair_fp32_uint64_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_parallel_sort_pair_fp64_uint64(daal::IdxValType<double> * begin_ptr, daal::IdxValType<double> * end_ptr)
{
    load_daal_thr_dll();
    static _daal_parallel_sort_pair_fp64_uint64_t _daal_parallel_sort_pair_fp64_uint64_ptr =
        (_daal_parallel_sort_pair_fp64_uint64_t)load_daal_thr_func("_daal_parallel_sort_pair_fp64_uint64");
    _daal_parallel_sort_pair_fp64_uint64_ptr(begin_ptr, end_ptr);
}

DAAL_EXPORT void _daal_threader_for_blocked(int n, int threads_request, const void * a, daal::functype2 func)
{
    load_daal_thr_dll();
    static _daal_threader_for_blocked_t _daal_threader_for_blocked_ptr =
        (_daal_threader_for_blocked_t)load_daal_thr_func("_daal_threader_for_blocked");
    _daal_threader_for_blocked_ptr(n, threads_request, a, func);
}

DAAL_EXPORT void _daal_threader_for_blocked_size(size_t n, size_t block, const void * a, daal::functype_blocked_size func)
{
    load_daal_thr_dll();
    static _daal_threader_for_blocked_size_t _daal_threader_for_blocked_size_ptr =
        (_daal_threader_for_blocked_size_t)load_daal_thr_func("_daal_threader_for_blocked_size");
    _daal_threader_for_blocked_size_ptr(n, block, a, func);
}

DAAL_EXPORT void _daal_threader_for_optional(int n, int threads_request, const void * a, daal::functype func)
{
    load_daal_thr_dll();
    static _daal_threader_for_t _daal_threader_for_optional_ptr = (_daal_threader_for_t)load_daal_thr_func("_daal_threader_for_optional");
    _daal_threader_for_optional_ptr(n, threads_request, a, func);
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                      const void * b, daal::reduction_functype_int64 reduction_func)
{
    load_daal_thr_dll();
    static _daal_parallel_reduce_int32_int64_t _daal_parallel_reduce_int32_int64_ptr =
        (_daal_parallel_reduce_int32_int64_t)load_daal_thr_func("_daal_parallel_reduce_int32_int64");
    return _daal_parallel_reduce_int32_int64_ptr(n, init, a, loop_func, b, reduction_func);
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32_int64_simple(int32_t n, int64_t init, const void * a, daal::loop_functype_int32_int64 loop_func,
                                                             const void * b, daal::reduction_functype_int64 reduction_func)
{
    load_daal_thr_dll();
    static _daal_parallel_reduce_int32_int64_t_simple _daal_parallel_reduce_int32_int64_simple_ptr =
        (_daal_parallel_reduce_int32_int64_t_simple)load_daal_thr_func("_daal_parallel_reduce_int32_int64_simple");
    return _daal_parallel_reduce_int32_int64_simple_ptr(n, init, a, loop_func, b, reduction_func);
}

DAAL_EXPORT int64_t _daal_parallel_reduce_int32ptr_int64_simple(const int32_t * begin, const int32_t * end, int64_t init, const void * a,
                                                                daal::loop_functype_int32ptr_int64 loop_func, const void * b,
                                                                daal::reduction_functype_int64 reduction_func)
{
    load_daal_thr_dll();
    static _daal_parallel_reduce_int32ptr_int64_t_simple _daal_parallel_reduce_int32ptr_int64_simple_ptr =
        (_daal_parallel_reduce_int32ptr_int64_t_simple)load_daal_thr_func("_daal_parallel_reduce_int32ptr_int64_simple");
    return _daal_parallel_reduce_int32ptr_int64_simple_ptr(begin, end, init, a, loop_func, b, reduction_func);
}

DAAL_EXPORT void _daal_threader_for_break(int n, int threads_request, const void * a, daal::functype_break func)
{
    load_daal_thr_dll();
    static _daal_threader_for_break_t _daal_threader_for_break_ptr = (_daal_threader_for_break_t)load_daal_thr_func("_daal_threader_for_break");
    _daal_threader_for_break_ptr(n, threads_request, a, func);
}

DAAL_EXPORT int _daal_threader_get_max_threads()
{
    load_daal_thr_dll();
    static _daal_threader_get_max_threads_t _daal_threader_get_max_threads_ptr =
        (_daal_threader_get_max_threads_t)load_daal_thr_func("_daal_threader_get_max_threads");
    return _daal_threader_get_max_threads_ptr();
}

DAAL_EXPORT int _daal_threader_get_current_thread_index()
{
    load_daal_thr_dll();
    static _daal_threader_get_current_thread_index_t _daal_threader_get_current_thread_index_ptr =
        (_daal_threader_get_current_thread_index_t)load_daal_thr_func("_daal_threader_get_current_thread_index");
    return _daal_threader_get_current_thread_index_ptr();
}

DAAL_EXPORT void * _daal_get_tls_ptr(void * a, daal::tls_functype func)
{
    load_daal_thr_dll();
    static _daal_get_tls_ptr_t _daal_get_tls_ptr_ptr = (_daal_get_tls_ptr_t)load_daal_thr_func("_daal_get_tls_ptr");
    return _daal_get_tls_ptr_ptr(a, func);
}

DAAL_EXPORT void _daal_del_tls_ptr(void * tlsPtr)
{
    load_daal_thr_dll();
    static _daal_del_tls_ptr_t _daal_del_tls_ptr_ptr = (_daal_del_tls_ptr_t)load_daal_thr_func("_daal_del_tls_ptr");
    _daal_del_tls_ptr_ptr(tlsPtr);
}

DAAL_EXPORT void * _daal_get_tls_local(void * tlsPtr)
{
    load_daal_thr_dll();
    static _daal_get_tls_local_t _daal_get_tls_local_ptr = (_daal_get_tls_local_t)load_daal_thr_func("_daal_get_tls_local");
    return _daal_get_tls_local_ptr(tlsPtr);
}

DAAL_EXPORT void _daal_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func)
{
    load_daal_thr_dll();
    static _daal_reduce_tls_t _daal_reduce_tls_ptr = (_daal_reduce_tls_t)load_daal_thr_func("_daal_reduce_tls");
    _daal_reduce_tls_ptr(tlsPtr, a, func);
}

DAAL_EXPORT void _daal_parallel_reduce_tls(void * tlsPtr, void * a, daal::tls_reduce_functype func)
{
    load_daal_thr_dll();
    static _daal_parallel_reduce_tls_t _daal_parallel_reduce_tls_ptr = (_daal_parallel_reduce_tls_t)load_daal_thr_func("_daal_parallel_reduce_tls");
    _daal_parallel_reduce_tls_ptr(tlsPtr, a, func);
}

DAAL_EXPORT void * _daal_get_ls_ptr(void * a, daal::tls_functype func)
{
    load_daal_thr_dll();
    static _daal_get_ls_ptr_t _daal_get_ls_ptr_ptr = (_daal_get_ls_ptr_t)load_daal_thr_func("_daal_get_ls_ptr");
    return _daal_get_ls_ptr_ptr(a, func);
}

DAAL_EXPORT void _daal_del_ls_ptr(void * lsPtr)
{
    load_daal_thr_dll();
    static _daal_del_ls_ptr_t _daal_del_ls_ptr_ptr = (_daal_del_ls_ptr_t)load_daal_thr_func("_daal_del_ls_ptr");
    _daal_del_ls_ptr_ptr(lsPtr);
}

DAAL_EXPORT void * _daal_get_ls_local(void * lsPtr)
{
    load_daal_thr_dll();
    static _daal_get_ls_local_t _daal_get_ls_local_ptr = (_daal_get_ls_local_t)load_daal_thr_func("_daal_get_ls_local");
    return _daal_get_ls_local_ptr(lsPtr);
}

DAAL_EXPORT void _daal_release_ls_local(void * lsPtr, void * a)
{
    load_daal_thr_dll();
    static _daal_release_ls_local_t _daal_release_ls_local_ptr = (_daal_release_ls_local_t)load_daal_thr_func("_daal_release_ls_local");
    _daal_release_ls_local_ptr(lsPtr, a);
}

DAAL_EXPORT void _daal_reduce_ls(void * lsPtr, void * a, daal::tls_reduce_functype func)
{
    load_daal_thr_dll();
    static _daal_reduce_ls_t _daal_reduce_ls_ptr = (_daal_reduce_ls_t)load_daal_thr_func("_daal_reduce_ls");
    _daal_reduce_ls_ptr(lsPtr, a, func);
}

DAAL_EXPORT void * _daal_new_mutex()
{
    load_daal_thr_dll();
    static _daal_new_mutex_t _daal_new_mutex_ptr = (_daal_new_mutex_t)load_daal_thr_func("_daal_new_mutex");
    return _daal_new_mutex_ptr();
}

DAAL_EXPORT void _daal_lock_mutex(void * mutexPtr)
{
    load_daal_thr_dll();
    static _daal_lock_mutex_t _daal_lock_mutex_ptr = (_daal_lock_mutex_t)load_daal_thr_func("_daal_lock_mutex");
    _daal_lock_mutex_ptr(mutexPtr);
}

DAAL_EXPORT void _daal_unlock_mutex(void * mutexPtr)
{
    load_daal_thr_dll();
    static _daal_unlock_mutex_t _daal_unlock_mutex_ptr = (_daal_unlock_mutex_t)load_daal_thr_func("_daal_unlock_mutex");
    _daal_unlock_mutex_ptr(mutexPtr);
}

DAAL_EXPORT void _daal_del_mutex(void * mutexPtr)
{
    load_daal_thr_dll();
    static _daal_del_mutex_t _daal_del_mutex_ptr = (_daal_del_mutex_t)load_daal_thr_func("_daal_del_mutex");
    _daal_del_mutex_ptr(mutexPtr);
}

DAAL_EXPORT void * _daal_new_task_group()
{
    load_daal_thr_dll();
    static _daal_new_task_group_t _daal_new_task_group_ptr = (_daal_new_task_group_t)load_daal_thr_func("_daal_new_task_group");
    return _daal_new_task_group_ptr();
}

DAAL_EXPORT void _daal_del_task_group(void * taskGroupPtr)
{
    load_daal_thr_dll();
    static _daal_del_task_group_t _daal_del_task_group_ptr = (_daal_del_task_group_t)load_daal_thr_func("_daal_del_task_group");
    _daal_del_task_group_ptr(taskGroupPtr);
}

DAAL_EXPORT void _daal_run_task_group(void * taskGroupPtr, daal::task * t)
{
    load_daal_thr_dll();
    static _daal_run_task_group_t _daal_run_task_group_ptr = (_daal_run_task_group_t)load_daal_thr_func("_daal_run_task_group");
    _daal_run_task_group_ptr(taskGroupPtr, t);
}

DAAL_EXPORT void _daal_wait_task_group(void * taskGroupPtr)
{
    load_daal_thr_dll();
    static _daal_wait_task_group_t _daal_wait_task_group_ptr = (_daal_wait_task_group_t)load_daal_thr_func("_daal_wait_task_group");
    _daal_wait_task_group_ptr(taskGroupPtr);
}

DAAL_EXPORT bool _daal_is_in_parallel()
{
    load_daal_thr_dll();
    static _daal_is_in_parallel_t _daal_is_in_parallel_ptr = (_daal_is_in_parallel_t)load_daal_thr_func("_daal_is_in_parallel");
    return _daal_is_in_parallel_ptr();
}

DAAL_EXPORT void _daal_tbb_task_scheduler_free(void *& init)
{
    if (init == NULL)
    {
        // If threading library was not opened, there is nothing to free,
        // so we do not need to load threading library.
        // Moreover, loading threading library in the Environment destructor
        // results in a crush because of the use of Wintrust library after it was unloaded.
        // This happens due to undefined order of static objects deinitialization
        // like Environment, and dependent libraries.
        return;
    }

    load_daal_thr_dll();
    static _daal_tbb_task_scheduler_free_t _daal_tbb_task_scheduler_free_ptr =
        (_daal_tbb_task_scheduler_free_t)load_daal_thr_func("_daal_tbb_task_scheduler_free");
    return _daal_tbb_task_scheduler_free_ptr(init);
}

DAAL_EXPORT void _daal_tbb_task_scheduler_handle_free(void *& init)
{
    load_daal_thr_dll();
    static _daal_tbb_task_scheduler_handle_free_t _daal_tbb_task_scheduler_handle_free_ptr =
        (_daal_tbb_task_scheduler_handle_free_t)load_daal_thr_func("_daal_tbb_task_scheduler_handle_free");
    return _daal_tbb_task_scheduler_handle_free_ptr(init);
}

DAAL_EXPORT size_t _setNumberOfThreads(const size_t numThreads, void ** init)
{
    load_daal_thr_dll();
    static _setNumberOfThreads_t _setNumberOfThreadsPtr = (_setNumberOfThreads_t)load_daal_thr_func("_setNumberOfThreads");
    return _setNumberOfThreadsPtr(numThreads, init);
}

DAAL_EXPORT size_t _setSchedulerHandle(void ** init)
{
    load_daal_thr_dll();
    static _setSchedulerHandle_t _setSchedulerHandlePtr = (_setSchedulerHandle_t)load_daal_thr_func("_setSchedulerHandle");
    return _setSchedulerHandlePtr(init);
}

DAAL_EXPORT void * _daal_threader_env()
{
    load_daal_thr_dll();
    static _daal_threader_env_t _daal_threader_env_ptr = (_daal_threader_env_t)load_daal_thr_func("_daal_threader_env");
    return _daal_threader_env_ptr();
}

#if !(defined DAAL_THREAD_PINNING_DISABLED)
DAAL_EXPORT void _thread_pinner_thread_pinner_init()
{
    load_daal_thr_dll();
    static _thread_pinner_thread_pinner_init_t _thread_pinner_thread_pinner_init_ptr =
        (_thread_pinner_thread_pinner_init_t)load_daal_thr_func("_thread_pinner_thread_pinner_init");
    _thread_pinner_thread_pinner_init_ptr();
}

DAAL_EXPORT void _thread_pinner_read_topology()
{
    load_daal_thr_dll();
    static _thread_pinner_read_topology_t _thread_pinner_read_topology_ptr =
        (_thread_pinner_read_topology_t)load_daal_thr_func("_thread_pinner_read_topology");
    _thread_pinner_read_topology_ptr();
}

DAAL_EXPORT void _thread_pinner_on_scheduler_entry(bool p)
{
    load_daal_thr_dll();
    static _thread_pinner_on_scheduler_entry_t _thread_pinner_on_scheduler_entry_ptr =
        (_thread_pinner_on_scheduler_entry_t)load_daal_thr_func("_thread_pinner_on_scheduler_entry");
    _thread_pinner_on_scheduler_entry_ptr(p);
}

DAAL_EXPORT void _thread_pinner_on_scheduler_exit(bool p)
{
    load_daal_thr_dll();
    static _thread_pinner_on_scheduler_exit_t _thread_pinner_on_scheduler_exit_ptr =
        (_thread_pinner_on_scheduler_exit_t)load_daal_thr_func("_thread_pinner_on_scheduler_exit");
    _thread_pinner_on_scheduler_exit_ptr(p);
}

DAAL_EXPORT void _thread_pinner_execute(daal::services::internal::thread_pinner_task_t & task)
{
    load_daal_thr_dll();
    static _thread_pinner_execute_t _thread_pinner_execute_ptr = (_thread_pinner_execute_t)load_daal_thr_func("_thread_pinner_execute");
    _thread_pinner_execute_ptr(task);
}

DAAL_EXPORT int _thread_pinner_get_status()
{
    load_daal_thr_dll();
    static _thread_pinner_get_status_t _thread_pinner_get_status_ptr = (_thread_pinner_get_status_t)load_daal_thr_func("_thread_pinner_get_status");
    return _thread_pinner_get_status_ptr();
}

DAAL_EXPORT bool _thread_pinner_get_pinning()
{
    load_daal_thr_dll();
    static _thread_pinner_get_pinning_t _thread_pinner_get_pinning_ptr =
        (_thread_pinner_get_pinning_t)load_daal_thr_func("_thread_pinner_get_pinning");
    return _thread_pinner_get_pinning_ptr();
}

DAAL_EXPORT bool _thread_pinner_set_pinning(bool p)
{
    load_daal_thr_dll();
    static _thread_pinner_set_pinning_t _thread_pinner_set_pinning_ptr =
        (_thread_pinner_set_pinning_t)load_daal_thr_func("_thread_pinner_set_pinning");
    return _thread_pinner_set_pinning_ptr(p);
}

DAAL_EXPORT void * _getThreadPinner(bool create_pinner, void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *))
{
    load_daal_thr_dll();
    static _getThreadPinner_t _getThreadPinnerPtr = (_getThreadPinner_t)load_daal_thr_func("_getThreadPinner");
    return _getThreadPinnerPtr(create_pinner, read_topo, deleter);
}
#endif // !(defined DAAL_THREAD_PINNING_DISABLED)

DAAL_EXPORT void _daal_threader_reduce(const size_t n, const size_t grainSize, daal::Reducer & reducer)
{
    load_daal_thr_dll();
    static _daal_threader_reduce_t _daal_threader_reduce_ptr = (_daal_threader_reduce_t)load_daal_thr_func("_daal_threader_reduce");
    _daal_threader_reduce_ptr(n, grainSize, reducer);
}

DAAL_EXPORT void _daal_static_threader_reduce(const size_t n, const size_t grainSize, daal::Reducer & reducer)
{
    load_daal_thr_dll();
    static _daal_static_threader_reduce_t _daal_static_threader_reduce_ptr =
        (_daal_static_threader_reduce_t)load_daal_thr_func("_daal_static_threader_reduce");
    _daal_static_threader_reduce_ptr(n, grainSize, reducer);
}
