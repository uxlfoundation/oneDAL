/* file: mt19937_batch_impl.h */
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
//  Implementation of the class defining the mt19937 engine
//--
*/

#include "algorithms/engines/mt19937/mt19937.h"
#include "src/algorithms/engines/engine_batch_impl.h"
#include "src/externals/service_rng.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mt19937
{
namespace internal
{
template <CpuType cpu, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class BatchImpl : public algorithms::engines::mt19937::interface1::Batch<algorithmFPType, method>, public algorithms::engines::internal::BatchBaseImpl
{
public:
    typedef algorithms::engines::mt19937::interface1::Batch<algorithmFPType, method> super1;
    typedef algorithms::engines::internal::BatchBaseImpl super2;
    BatchImpl(size_t seed = 777) : baseRng(seed, __DAAL_BRNG_MT19937), super2(seed) {}

    void * getState() override { return baseRng.getState(); }

    int getStateSize() const override { return baseRng.getStateSize(); }

    services::Status saveStateImpl(byte * dest) const override
    {
        DAAL_CHECK(!baseRng.saveState((void *)dest), ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

    services::Status loadStateImpl(const byte * src) override
    {
        DAAL_CHECK(!baseRng.loadState((const void *)src), ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

    services::Status leapfrogImpl(size_t threadNum, size_t nThreads) override
    {
        int errcode = baseRng.leapfrog(threadNum, nThreads);
        services::Status s;
        if (errcode == __DAAL_LEAPFROG_METHOD_ERRCODE)
            s.add(ErrorLeapfrogUnsupported);
        else if (errcode)
            s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    services::Status skipAheadImpl(size_t nSkip) override
    {
        int errcode = baseRng.skipAhead(nSkip);
        services::Status s;
        if (errcode == __DAAL_SKIP_AHEAD_METHOD_ERRCODE)
            s.add(ErrorSkipAheadUnsupported);
        else if (errcode)
            s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    virtual BatchImpl<cpu, algorithmFPType, method> * cloneImpl() const override { return new BatchImpl<cpu, algorithmFPType, method>(*this); }

    bool hasSupport(engines::internal::ParallelizationTechnique technique) const override
    {
        switch (technique)
        {
        case engines::internal::family: return false;
        case engines::internal::skipahead: return true;
        case engines::internal::leapfrog: return false;
        }
        return false;
    }

    ~BatchImpl() {}

protected:
    BatchImpl(const BatchImpl<cpu, algorithmFPType, method> & other) : super1(other), super2(other), baseRng(other.baseRng) {}
    daal::internal::BaseRNGsInst<cpu> baseRng;
};

} // namespace internal
} // namespace mt19937
} // namespace engines
} // namespace algorithms
} // namespace daal
