/* file: mrg32k3a_batch_impl.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
//  Implementation of the class defining the mrg32k3a engine.
//--
*/

#include "algorithms/engines/mrg32k3a/mrg32k3a.h"
#include "src/algorithms/engines/engine_batch_impl.h"
#include "src/externals/service_rng.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mrg32k3a
{
namespace internal
{
template <CpuType cpu, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class BatchImpl : public algorithms::engines::mrg32k3a::interface1::Batch<algorithmFPType, method>,
                  public algorithms::engines::internal::BatchBaseImpl
{
public:
    typedef algorithms::engines::mrg32k3a::interface1::Batch<algorithmFPType, method> super1;
    typedef algorithms::engines::internal::BatchBaseImpl super2;
    BatchImpl(size_t seed = 777) : baseRng(seed, __DAAL_BRNG_MRG32K3A), super2(seed) {}

    void * getState() DAAL_C11_OVERRIDE { return baseRng.getState(); }

    int getStateSize() const DAAL_C11_OVERRIDE { return baseRng.getStateSize(); }

    services::Status saveStateImpl(byte * dest) const DAAL_C11_OVERRIDE
    {
        DAAL_CHECK(!baseRng.saveState((void *)dest), ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

    services::Status loadStateImpl(const byte * src) DAAL_C11_OVERRIDE
    {
        DAAL_CHECK(!baseRng.loadState((const void *)src), ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

    services::Status leapfrogImpl(size_t threadNum, size_t nThreads) DAAL_C11_OVERRIDE
    {
        int errcode = baseRng.leapfrog(threadNum, nThreads);
        services::Status s;
        if (errcode == __DAAL_RNG_ERROR_LEAPFROG_UNSUPPORTED)
            s.add(ErrorLeapfrogUnsupported);
        else if (errcode)
            s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    services::Status skipAheadImpl(size_t nSkip) DAAL_C11_OVERRIDE
    {
        int errcode = baseRng.skipAhead(nSkip);
        services::Status s;
        if (errcode == __DAAL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED)
            s.add(ErrorSkipAheadUnsupported);
        else if (errcode)
            s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    virtual BatchImpl<cpu, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new BatchImpl<cpu, algorithmFPType, method>(*this);
    }

    bool hasSupport(engines::internal::ParallelizationTechnique technique) const DAAL_C11_OVERRIDE
    {
        switch (technique)
        {
        case engines::internal::family: return false;
        case engines::internal::skipahead: return true;
        case engines::internal::leapfrog: return true;
        }
        return false;
    }

    ~BatchImpl() {}

protected:
    BatchImpl(const BatchImpl<cpu, algorithmFPType, method> & other) : super1(other), super2(other), baseRng(other.baseRng) {}

    daal::internal::BaseRNGsInst<cpu> baseRng;
};

} // namespace internal
} // namespace mrg32k3a
} // namespace engines
} // namespace algorithms
} // namespace daal
