/* file: service_thread_pinner.cpp */
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
//  Implementation of thread pinner class
//--
*/
#include "services/daal_defines.h"
#if !(defined DAAL_THREAD_PINNING_DISABLED)

    #include "src/threading/service_thread_pinner.h"
    #include "services/daal_memory.h"
    #include "src/threading/threading.h"

    #define USE_TASK_ARENA_CURRENT_SLOT 1
    #define LOG_PINNING                 1
    #define TBB_PREVIEW_TASK_ARENA      1
    #define TBB_PREVIEW_LOCAL_OBSERVER  1

    #include "tbb/tbb.h"
    #include <tbb/task_arena.h>
    #include <tbb/task_scheduler_observer.h>
    #include <tbb/parallel_reduce.h>
    #include <tbb/blocked_range.h>
    #include <tbb/tick_count.h>
    #include <tbb/scalable_allocator.h>
    #include "services/daal_atomic_int.h"
using namespace daal::services;

    #if defined(_WIN32) || defined(_WIN64)
        #include <windows.h>
        #define __PINNER_WINDOWS__

        #if defined(_WIN64)
            #define MASK_WIDTH 64
        #else
            #define MASK_WIDTH 32
        #endif

    #else // LINUX
        #include <sched.h>
        #define __PINNER_LINUX__

        #ifdef __FreeBSD__
            #include <pthread_np.h>

cpu_set_t * __sched_cpualloc(size_t count)
{
    return (cpu_set_t *)malloc(CPU_ALLOC_SIZE(count));
}
void __sched_cpufree(cpu_set_t * set)
{
    free(set);
}
int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t * mask)
{
    return cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid == 0 ? -1 : pid, cpusetsize, mask);
}
        #endif

    #endif

struct cpu_mask_t
{
    int status;
    #if defined(_WIN32) || defined(_WIN64)
    GROUP_AFFINITY ga;
    #else
    int ncpus;
    int bit_parts_size;
    cpu_set_t * cpu_set;
    #endif
    cpu_mask_t()
    {
        status = 0;

    #if defined __PINNER_LINUX__

        ncpus          = 0;
        bit_parts_size = 0;
        cpu_set        = NULL;
        for (ncpus = sizeof(cpu_set_t) / CHAR_BIT; ncpus < 16 * 1024; ncpus <<= 1)
        {
            cpu_set = CPU_ALLOC(ncpus);
            if (cpu_set == NULL) break;

            bit_parts_size = CPU_ALLOC_SIZE(ncpus);
            CPU_ZERO_S(bit_parts_size, cpu_set);

            const int err = sched_getaffinity(0, bit_parts_size, cpu_set);
            if (err == 0) break;

            CPU_FREE(cpu_set);
            cpu_set = NULL;
            if (errno != EINVAL) break;
        }

        if (cpu_set == NULL)
    #else // defined __PINNER_WINDOWS__
        bool retval = GetThreadGroupAffinity(GetCurrentThread(), &ga);
        if (!retval)
    #endif
        {
            status--;
        }

        return;
    }

    int get_thread_affinity()
    {
        if (status == 0)
        {
    #if defined __PINNER_LINUX__
            int err = pthread_getaffinity_np(pthread_self(), bit_parts_size, cpu_set);
            if (err)
    #else // defined __PINNER_WINDOWS__
            bool retval = GetThreadGroupAffinity(GetCurrentThread(), &ga);
            if (!retval)
    #endif
            {
                status--;
            }
        }

        return status;
    } // int get_thread_affinity()

    int set_thread_affinity()
    {
        if (status == 0)
        {
    #if defined __PINNER_LINUX__

            int err = pthread_setaffinity_np(pthread_self(), bit_parts_size, cpu_set);
            if (err)
    #else // defined __PINNER_WINDOWS__

            bool retval = SetThreadGroupAffinity(GetCurrentThread(), &ga, NULL);
            if (!retval)
    #endif
            {
                status--;
            }
        }

        return status;
    } // int set_thread_affinity()

    int set_cpu_index(int cpu_idx)
    {
        if (status == 0)
        {
    #if defined __PINNER_LINUX__
            CPU_ZERO_S(bit_parts_size, cpu_set);
            CPU_SET_S(cpu_idx, bit_parts_size, cpu_set);
    #else // defined __PINNER_WINDOWS__
            ga.Group = cpu_idx / MASK_WIDTH;
            ga.Mask  = cpu_idx % MASK_WIDTH;
    #endif
        }

        return status;
    } // int set_cpu_index(int cpu_idx)

    int get_status() { return status; } // int get_status()

    ~cpu_mask_t()
    {
    #if defined __PINNER_LINUX__
        if (cpu_set != NULL)
        {
            CPU_FREE(cpu_set);
        }
    #endif

        return;
    } // ~cpu_mask_t()
};

class thread_pinner_impl_t : public tbb::task_scheduler_observer
{
    int status;
    int nthreads;
    int max_threads;
    int * cpu_queue;
    bool do_pinning;
    AtomicInt is_pinning;
    tbb::enumerable_thread_specific<cpu_mask_t *> thread_mask;
    tbb::task_arena pinner_arena;
    void (*topo_deleter)(void *);

public:
    thread_pinner_impl_t(void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *));
    void on_scheduler_entry(bool);
    void on_scheduler_exit(bool);
    void init_thread_pinner(int statusToSet, int nthreadsToSet, int max_threadsToSet, int * cpu_queueToSet);

    void execute(daal::services::internal::thread_pinner_task_t & task)
    {
        if (do_pinning && (status == 0) && (is_pinning.get() == 0))
        {
            is_pinning.set(1);
            pinner_arena.execute(task);
            is_pinning.set(0);
        }
        else
        {
            task();
        }
    }

    int get_status();
    bool get_pinning();
    bool set_pinning(bool p);
    ~thread_pinner_impl_t();
} * IMPL;

thread_pinner_impl_t::thread_pinner_impl_t(void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *))
    : pinner_arena(nthreads = daal::threader_get_threads_number()), tbb::task_scheduler_observer(pinner_arena), topo_deleter(deleter)
{
    #if defined(TARGET_X86_64)
    pinner_arena.initialize();
    #endif
    do_pinning = (nthreads > 0) ? true : false;
    is_pinning.set(0);

    read_topo(status, nthreads, max_threads, &cpu_queue);
    observe(true);

    return;
} /* thread_pinner_impl_t() */

void thread_pinner_impl_t::on_scheduler_entry(bool) /*override*/
{
    if (do_pinning == false || status < 0) return;

    // read current thread index
    const int thr_idx = tbb::this_task_arena::current_thread_index();

    // Get next cpu from topology queue
    int cpu_idx = cpu_queue[thr_idx % max_threads];

    // Allocate source and target affinity masks
    cpu_mask_t * target_mask = new cpu_mask_t;
    cpu_mask_t * source_mask = thread_mask.local();

    // Create source mask if it wasn't created for the tread before
    if (source_mask == NULL)
    {
        source_mask         = new cpu_mask_t();
        thread_mask.local() = source_mask;
    }

    // save source affinity mask to restore on exit
    status -= source_mask->get_thread_affinity();

    // Set ine bit corresponding to CPU to pin the thread
    status -= target_mask->set_cpu_index(cpu_idx);

    // Set thread affinity mask to 1 non-zero bit in corresponding to cpu_idx position
    status -= target_mask->set_thread_affinity();

    delete target_mask;

    return;
} /* void on_scheduler_entry()  */

void thread_pinner_impl_t::on_scheduler_exit(bool) /*override*/
{
    if (do_pinning == false || status < 0) return;

    // get current thread original mask
    cpu_mask_t * source_mask = thread_mask.local();

    if (source_mask == NULL)
    {
        status--;
        return;
    }
    else
    {
        // restore original thread affinity mask
        status -= source_mask->set_thread_affinity();
        if (status < 0)
        {
            status--;
            return;
        }
    }

    return;
} /* void on_scheduler_exit( bool ) */

int thread_pinner_impl_t::get_status()
{
    return status;
} /* int get_status() */

bool thread_pinner_impl_t::get_pinning()
{
    return do_pinning;
} /* bool get_pinning() */

bool thread_pinner_impl_t::set_pinning(bool p)
{
    bool old_pinning = do_pinning;
    if (status == 0) do_pinning = p;

    return old_pinning;
} /* bool set_pinning(bool p) */

thread_pinner_impl_t::~thread_pinner_impl_t()
{
    observe(false);

    if (cpu_queue) topo_deleter(cpu_queue);

    thread_mask.combine_each([](cpu_mask_t *& source_mask) { delete source_mask; });

    return;
} /* ~thread_pinner_impl_t() */

DAAL_EXPORT void * _getThreadPinner(bool create_pinner, void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *))
{
    static bool pinner_created = false;

    if (create_pinner == true || pinner_created == true)
    {
        static daal::services::internal::thread_pinner_t thread_pinner(read_topo, deleter);
        if (thread_pinner.get_status() == 0)
        {
            pinner_created = true;
            return (void *)&thread_pinner;
        }
    }

    return NULL;
} /* thread_pinner_t* getThreadPinner() */

DAAL_EXPORT void _thread_pinner_thread_pinner_init(void (*read_topo)(int &, int &, int &, int **), void (*deleter)(void *))
{
    static thread_pinner_impl_t impl(read_topo, deleter);
    IMPL = &impl;
}

DAAL_EXPORT void _thread_pinner_execute(daal::services::internal::thread_pinner_task_t & task)
{
    IMPL->execute(task);
}

DAAL_EXPORT int _thread_pinner_get_status()
{
    return IMPL->get_status();
}

DAAL_EXPORT bool _thread_pinner_get_pinning()
{
    return IMPL->get_pinning();
}

DAAL_EXPORT bool _thread_pinner_set_pinning(bool p)
{
    return IMPL->set_pinning(p);
}

DAAL_EXPORT void _thread_pinner_on_scheduler_entry(bool p)
{
    IMPL->on_scheduler_entry(p);
}

DAAL_EXPORT void _thread_pinner_on_scheduler_exit(bool p)
{
    IMPL->on_scheduler_exit(p);
}

#endif /* #if !defined (DAAL_THREAD_PINNING_DISABLED) */
