/* file: service_thread_declar_ref.h */
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
//  Auxiliary class to set/restore OpenBLAS threads
//--
*/

#ifndef __SERVICE_THREAD_DECLAR_REF_H__
#define __SERVICE_THREAD_DECLAR_REF_H__

#include <mutex>

namespace daal
{
namespace internal
{
namespace ref
{
extern "C"
{
    extern void openblas_set_num_threads(int num_threads);
    extern int openblas_get_num_threads(void);
}

/* OpenBLAS lacks a thread-local equivalent to mkl_set_num_threads_local(),
   so openblas_set_num_threads() modifies global state. A static mutex
   serializes the save-set-restore sequence to prevent race conditions
   when multiple threads enter single-threaded BLAS/LAPACK calls concurrently.
   See: https://github.com/uxlfoundation/oneDAL/issues/3329 */
class openblas_thread_setter
{
public:
    openblas_thread_setter(int n_threads = 1)
    {
        mutex().lock();
        previous_thread_count = openblas_get_num_threads();
        openblas_set_num_threads(n_threads);
    }
    ~openblas_thread_setter()
    {
        openblas_set_num_threads(previous_thread_count);
        mutex().unlock();
    }

    openblas_thread_setter(const openblas_thread_setter &)             = delete;
    openblas_thread_setter & operator=(const openblas_thread_setter &) = delete;

private:
    int previous_thread_count = 1;

    static std::mutex & mutex()
    {
        static std::mutex m;
        return m;
    }
};

} // namespace ref
} // namespace internal
} // namespace daal

#endif
