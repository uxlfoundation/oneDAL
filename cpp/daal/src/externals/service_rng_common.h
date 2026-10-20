/* file: service_rng_common.h */
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
//  Common RNG classes.
//--
*/

#ifndef __SERVICE_RNG_COMMON_H__
#define __SERVICE_RNG_COMMON_H__

#include <cstddef>

#include "services/daal_defines.h"

namespace daal
{
namespace internal
{
// Largest skip the one-word skip-ahead takes: its count is a long long.
constexpr DAAL_UINT64 maxOneWordSkip = 0x7FFFFFFFFFFFFFFFull;

/// Whether a multi-word skip nSkip[0] + nSkip[1] * 2^64 + ... fits a one-word skip-ahead.
inline bool fitsOneWord(size_t nWords, const DAAL_UINT64 * nSkip)
{
    for (size_t i = 1; i < nWords; ++i)
        if (nSkip[i]) return false;
    return !nWords || nSkip[0] <= maxOneWordSkip;
}

template <CpuType cpu>
class BaseRNGIface
{
public:
    virtual int getStateSize() const                        = 0;
    virtual int saveState(void * dest) const                = 0;
    virtual int loadState(const void * src)                 = 0;
    virtual int leapfrog(size_t threadNum, size_t nThreads) = 0;
    virtual int skipAhead(size_t nSkip)                     = 0;
    virtual void * getState()                               = 0;
    /// Skips nSkip[0] + nSkip[1] * 2^64 + ... outputs; nSkip has nWords 64-bit words. Counter-addressed
    /// draws (dtrees/engine/node_rng.h) reach positions beyond 2^64; backends that cannot return an error.
    virtual int skipAheadEx(size_t nWords, const DAAL_UINT64 * nSkip) = 0;
    /// Copies the state of a stream of the same generator, given by its getState(), without allocating,
    /// so a thread can reset its own stream before each counter-addressed read.
    virtual int copyState(void * srcState) = 0;
};

} // namespace internal
} // namespace daal

#endif
