/* file: mt2203_batch_impl.h */
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
//  Implementation of the class defining the mt2203 engine
//--
*/

#include "algorithms/engines/mt2203/mt2203.h"
#include "src/algorithms/engines/engine_batch_impl.h"
#include "src/externals/service_rng.h"
#include "src/data_management/service_numeric_table.h"
#include "services/collection.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mt2203
{
namespace internal
{
template <CpuType cpu, typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class BatchImpl : public algorithms::engines::mt2203::interface1::Batch<algorithmFPType, method>, public algorithms::engines::internal::BatchBaseImpl
{
private:
    using BaseRNGsPtr = SharedPtr<daal::internal::BaseRNGsInst<cpu> >;

private:
    const int32_t header = 0x4441414C;

public:
    typedef algorithms::engines::mt2203::interface1::Batch<algorithmFPType, method> super1;
    typedef algorithms::engines::internal::BatchBaseImpl super2;

    BatchImpl(size_t seed = 777, services::Status * st = nullptr) : super1(seed), super2(seed)
    {
        push_back(0, BaseRNGsPtr(new daal::internal::BaseRNGsInst<cpu>(seed, __DAAL_BRNG_MT2203)));
    }

    services::Status addImpl(size_t numberOfStreams) DAAL_C11_OVERRIDE
    {
        Collection<size_t> indices(numberOfStreams);
        services::Status s = requestIndices(numberOfStreams, 6024, indices);

        for (size_t i = 0; i < numberOfStreams; i++)
        {
            push_back(indices[i], BaseRNGsPtr(new daal::internal::BaseRNGsInst<cpu>(this->getSeed(), __DAAL_BRNG_MT2203 + indices[i])));
        }
        return s;
    }

    FamilyEnginePtr getImpl(size_t i) const DAAL_C11_OVERRIDE { return FamilyEnginePtr(new BatchImpl(this->getSeed(), _streamIdxs[i], _streams[i])); }

    void * getState() DAAL_C11_OVERRIDE { return _streams[0]->getState(); }

    int getStateSize() const DAAL_C11_OVERRIDE { return _streams[0]->getStateSize(); }

    services::Status saveStateImpl(byte * dest) const DAAL_C11_OVERRIDE
    {
        DAAL_CHECK(dest, ErrorIncorrectErrorcodeFromGenerator);

        int * headerIn = (int *)(dest);
        *headerIn      = header;

        services::Collection<size_t> * streamIdxsIn = new (dest + sizeof(int)) services::Collection<size_t>();
        services::Collection<BaseRNGsPtr> * streamsIn =
            new (dest + sizeof(int) + sizeof(services::Collection<size_t>)) services::Collection<BaseRNGsPtr>();

        streamIdxsIn->resize(this->getNumberOfStreams());
        streamsIn->resize(this->getNumberOfStreams());

        for (size_t i = 0; i < this->getNumberOfStreams(); i++)
        {
            (*streamIdxsIn)[i] = _streamIdxs[i];
        }

        for (size_t i = 0; i < this->getNumberOfStreams(); i++)
        {
            (*streamsIn)[i] = BaseRNGsPtr(new daal::internal::BaseRNGsInst<cpu>(*(_streams[i])));
        }
        return services::Status();
    }

    services::Status loadStateImpl(const byte * src) DAAL_C11_OVERRIDE
    {
        DAAL_CHECK(src, ErrorIncorrectErrorcodeFromGenerator);
        const int * headerIn = (const int *)(src);
        DAAL_CHECK(*headerIn == header, ErrorIncorrectErrorcodeFromGenerator);

        services::Collection<size_t> & streamIdxsIn = *(services::Collection<size_t> *)(src + sizeof(int));
        services::Collection<BaseRNGsPtr> & streamsIn =
            *(services::Collection<BaseRNGsPtr> *)(src + sizeof(int) + sizeof(services::Collection<size_t>));

        _streamIdxs.resize(this->getNumberOfStreams());
        _streams.resize(this->getNumberOfStreams());

        for (size_t i = 0; i < this->getNumberOfStreams(); i++)
        {
            _streamIdxs[i] = streamIdxsIn[i];
        }

        for (size_t i = 0; i < this->getNumberOfStreams(); i++)
        {
            _streams[i] = BaseRNGsPtr(new daal::internal::BaseRNGsInst<cpu>(*(streamsIn[i])));
        }

        return services::Status();
    }

    services::Status leapfrogImpl(size_t threadNum, size_t nThreads) DAAL_C11_OVERRIDE
    {
        Status s;
        for (size_t i = 0; i < this->getNumberOfStreams(); i++)
        {
            s.add(leapfrogOne(i, threadNum, nThreads));
        }
        return s;
    }

    services::Status skipAheadImpl(size_t nSkip) DAAL_C11_OVERRIDE
    {
        Status s;
        for (size_t i = 0; i < this->getNumberOfStreams(); i++)
        {
            s.add(skipAheadOne(i, nSkip));
        }
        return s;
    }

    virtual BatchImpl<cpu, algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new BatchImpl<cpu, algorithmFPType, method>(*this);
    }

    size_t getNumberOfStreamsImpl() const DAAL_C11_OVERRIDE { return _streams.size(); }

    size_t getMaxNumberOfStreamsImpl() const DAAL_C11_OVERRIDE { return 6024; }

    bool hasSupport(engines::internal::ParallelizationTechnique technique) const DAAL_C11_OVERRIDE
    {
        switch (technique)
        {
        case engines::internal::family: return true;
        case engines::internal::skipahead: return false;
        case engines::internal::leapfrog: return false;
        }
        return false;
    }

    ~BatchImpl() {}

protected:
    BatchImpl(const BatchImpl<cpu, algorithmFPType, method> & other) : super1(other), super2(other)
    {
        for (size_t i = 0; i < other.getNumberOfStreams(); i++)
        {
            push_back(other._streamIdxs[i], BaseRNGsPtr(new daal::internal::BaseRNGsInst<cpu>(*(other._streams[i]))));
        }
    }

    BatchImpl(size_t seed, size_t i, services::SharedPtr<daal::internal::BaseRNGsInst<cpu> > baseRng) : super1(seed), super2(seed)
    {
        push_back(i, baseRng);
    }

    services::Status leapfrogOne(size_t i, size_t threadNum, size_t nThreads)
    {
        int errcode = _streams[i]->leapfrog(threadNum, nThreads);
        services::Status s;
        if (errcode == __DAAL_RNG_ERROR_LEAPFROG_UNSUPPORTED)
            s.add(ErrorLeapfrogUnsupported);
        else if (errcode)
            s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    services::Status skipAheadOne(size_t i, size_t nSkip)
    {
        int errcode = _streams[i]->skipAhead(nSkip);
        services::Status s;
        if (errcode == __DAAL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED)
            s.add(ErrorSkipAheadUnsupported);
        else if (errcode)
            s.add(ErrorIncorrectErrorcodeFromGenerator);
        return s;
    }

    void push_back(size_t idx, SharedPtr<daal::internal::BaseRNGsInst<cpu> > brngPtr)
    {
        _streamIdxs.push_back(idx);
        _streams.push_back(brngPtr);
    }

    services::Status requestIndices(size_t nStreams, size_t maxStreams, Collection<size_t> & indices)
    {
        size_t i, idx = 0;
        for (i = 0; i < maxStreams; i++)
        {
            if (find(i) == -1)
            {
                indices[idx++] = i;
                if (idx == nStreams) break;
            }
        }
        if (i == maxStreams) return services::Status(ErrorIncorrectErrorcodeFromGenerator);
        return services::Status();
    }

    int find(size_t idx)
    {
        for (size_t i = 0; i < _streamIdxs.size(); i++)
        {
            if (_streamIdxs[i] == idx) return i;
        }
        return -1;
    }

    services::Collection<size_t> _streamIdxs;
    services::Collection<BaseRNGsPtr> _streams;
};

} // namespace internal
} // namespace mt2203
} // namespace engines
} // namespace algorithms
} // namespace daal
