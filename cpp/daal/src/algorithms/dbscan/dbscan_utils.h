/* file: dbscan_utils.h */
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
//  Common functions for DBSCAN
//--
*/

#ifndef __DBSCAN_IMPL_I__
#define __DBSCAN_IMPL_I__

#include "algorithms/dbscan/dbscan_types.h"

#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "services/error_handling.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_error_handling.h"

using namespace daal::internal;
using namespace daal::algorithms::internal;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace internal
{
#define __DBSCAN_DEFAULT_QUEUE_SIZE        32
#define __DBSCAN_DEFAULT_VECTOR_SIZE       32
#define __DBSCAN_DEFAULT_NEIGHBORHOOD_SIZE 64

template <typename T, CpuType cpu>
class Queue
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_QUEUE_SIZE;

public:
    Queue() : _data(nullptr), _head(0), _tail(0), _capacity(0) {}

    ~Queue() { clear(); }

    Queue(const Queue &)             = delete;
    Queue & operator=(const Queue &) = delete;

    void clear()
    {
        if (_data)
        {
            daal::services::internal::service_free<T, cpu>(_data);
            _data = nullptr;
        }
    }

    void reset() { _head = _tail = 0; }

    services::Status push(const T & value)
    {
        if (_tail >= _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }

        _data[_tail] = value;
        _tail++;

        return services::Status();
    }

    T pop()
    {
        if (_head < _tail)
        {
            const T value = _data[_head];
            _head++;
            return value;
        }

        return (T)0;
    }

    bool empty() const { return (_head == _tail); }
    size_t head() const { return _head; }
    size_t tail() const { return _tail; }

    T * getInternalPtr(size_t ind)
    {
        if (ind >= _tail)
        {
            return nullptr;
        }

        return &_data[ind];
    }

private:
    services::Status grow()
    {
        int result = 0;
        _capacity  = (_capacity == 0 ? defaultSize : _capacity * 2);

        T * const newData = daal::services::internal::service_malloc<T, cpu>(_capacity);
        DAAL_CHECK_MALLOC(newData);

        if (_data != nullptr)
        {
            result = services::internal::daal_memcpy_s(newData, _tail * sizeof(T), _data, _tail * sizeof(T));
            daal::services::internal::service_free<T, cpu>(_data);
            _data = nullptr;
        }

        _data = newData;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    T * _data;
    size_t _head;
    size_t _tail;
    size_t _capacity;
};

template <typename T, CpuType cpu>
class Vector
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_VECTOR_SIZE;

public:
    DAAL_NEW_DELETE();

    Vector() : _data(nullptr), _size(0), _capacity(0) {}

    ~Vector() { clear(); }

    Vector(const Vector &)             = delete;
    Vector & operator=(const Vector &) = delete;

    void clear()
    {
        if (_data)
        {
            services::internal::service_scalable_free<T, cpu>(_data);
            _data = nullptr;
        }
    }

    void reset() { _size = 0; }

    services::Status push_back(const T & value)
    {
        if (_size >= _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status);
        }

        _data[_size] = value;
        _size++;

        return services::Status();
    }

    /// Make sure `count` more values fit; write them through ptr() and then call resize().
    services::Status reserve(size_t count)
    {
        while (_size + count > _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status);
        }
        return services::Status();
    }

    void resize(size_t size) { _size = size; }

    inline T & operator[](size_t index) { return _data[index]; }

    size_t size() const { return _size; }

    T * ptr() { return _data; }

private:
    // The scalable allocator reuses freed blocks, so a buffer rebuilt for every call does not fault in fresh pages.
    services::Status grow()
    {
        int result            = 0;
        const size_t capacity = (_capacity == 0 ? defaultSize : _capacity * 2);
        T * const newData     = services::internal::service_scalable_malloc<T, cpu>(capacity);
        DAAL_CHECK_MALLOC(newData);

        if (_data != nullptr)
        {
            result = services::internal::daal_memcpy_s(newData, _size * sizeof(T), _data, _size * sizeof(T));
            services::internal::service_scalable_free<T, cpu>(_data);
        }

        _data     = newData;
        _capacity = capacity;
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    T * _data;
    size_t _size;
    size_t _capacity;
};

template <typename FPType, CpuType cpu>
class Neighborhood
{
    static const size_t defaultSize = __DBSCAN_DEFAULT_NEIGHBORHOOD_SIZE;

public:
    DAAL_NEW_DELETE();

    Neighborhood() : _values(nullptr), _capacity(0), _size(0), _weight(0) {}

    Neighborhood(const Neighborhood &)             = delete;
    Neighborhood & operator=(const Neighborhood &) = delete;

    ~Neighborhood() { clear(); }

    void clear()
    {
        if (_values)
        {
            daal::services::internal::service_scalable_free<size_t, cpu>(_values);
            _values = nullptr;
        }
        _capacity = _size = 0;
        _weight           = 0;
    }

    void reset()
    {
        _size   = 0;
        _weight = 0;
    }

    inline int add(const size_t & value, FPType w)
    {
        if (_size >= _capacity)
        {
            auto res = grow();
            if (res)
            {
                return res;
            }
        }

        _values[_size] = value;
        _size++;
        _weight += w;

        return 0;
    }

    void addWeight(FPType w) { _weight += w; }

    /// Make room for `count` more values with a single allocation.
    /// Up to 1024 values the capacity is exact; above that it is rounded up to a power of two,
    /// because odd sizes miss the scalable allocator's large-object cache.
    int reserve(size_t count)
    {
        const size_t required = _size + count;
        if (required <= _capacity)
        {
            return 0;
        }
        size_t capacity = required;
        if (capacity > 1024)
        {
            capacity = 1024;
            while (capacity < required) capacity *= 2;
        }
        return reallocate(capacity);
    }

    /// Append a value whose weight was already added with addWeight(); reserve() first.
    void push(size_t value) { _values[_size++] = value; }

    size_t get(size_t id) const { return _values[id]; }

    size_t size() const { return _size; }

    FPType weight() const { return _weight; }

private:
    int grow()
    {
        const size_t scale = _capacity < 128 ? 64 : 2;
        return reallocate(_capacity == 0 ? defaultSize : _capacity * scale);
    }

    /// Returns non-zero if the allocation or the copy fails; the neighborhood is unchanged then.
    int reallocate(size_t capacity)
    {
        size_t * const newValues = daal::services::internal::service_scalable_malloc<size_t, cpu>(capacity);
        if (!newValues)
        {
            return 1;
        }

        int result = 0;
        if (_values != nullptr)
        {
            result = services::internal::daal_memcpy_s(newValues, _size * sizeof(size_t), _values, _size * sizeof(size_t));
            daal::services::internal::service_scalable_free<size_t, cpu>(_values);
        }
        _values   = newValues;
        _capacity = capacity;

        return result;
    }

    size_t * _values;

    size_t _capacity;
    size_t _size;

    FPType _weight;
};

template <typename FPType, CpuType cpu>
struct TlsNTask
{
    TlsNTask(size_t n) { neighs = new Neighborhood<FPType, cpu>[n]; }

    ~TlsNTask()
    {
        if (neighs)
        {
            delete[] neighs;
            neighs = nullptr;
        }
    }

    static TlsNTask * create(size_t n)
    {
        TlsNTask<FPType, cpu> * result = new TlsNTask<FPType, cpu>(n);
        if (!result)
        {
            return nullptr;
        }
        if (!result->neighs)
        {
            delete result;
            result = nullptr;
            return nullptr;
        }
        return result;
    }

    Neighborhood<FPType, cpu> * neighs;
};

template <typename FPType, CpuType cpu>
struct NTask
{
    NTask(size_t _n)
    {
        n        = _n;
        tlsNTask = new daal::tls<TlsNTask<FPType, cpu> *>([=]() -> TlsNTask<FPType, cpu> * { return TlsNTask<FPType, cpu>::create(n); });
    }

    ~NTask()
    {
        if (tlsNTask)
        {
            tlsNTask->reduce([=](TlsNTask<FPType, cpu> * tt) -> void {
                delete tt;
                tt = nullptr;
            });
            delete tlsNTask;
        }
    }

    static services::SharedPtr<NTask<FPType, cpu> > create(size_t n)
    {
        services::SharedPtr<NTask<FPType, cpu> > result(new NTask<FPType, cpu>(n));
        if (result.get() && (!result->tlsNTask))
        {
            result.reset();
        }
        return result;
    }

    size_t n;
    daal::tls<TlsNTask<FPType, cpu> *> * tlsNTask;
};

/// Maps `eps` to the space the distances are compared in: squared for Euclidean, raised to `p`
/// for Minkowski, unchanged for the metrics that take no root.
///
/// @param[in] metric Distance metric
/// @param[in] eps    Neighborhood radius in the units of `metric`
/// @param[in] p      Minkowski degree; ignored by the other metrics
/// @return The threshold the unrooted distances are compared against
template <typename FPType, CpuType cpu>
inline FPType poweredEpsilon(PairwiseDistanceType metric, FPType eps, FPType p)
{
    switch (metric)
    {
    case PairwiseDistanceType::euclidean: return eps * eps;
    case PairwiseDistanceType::minkowski: return MathInst<FPType, cpu>::sPowx(eps, p);
    default: return eps;
    }
}

template <Method, typename FPType, CpuType cpu>
class NeighborhoodEngine
{
public:
    NeighborhoodEngine(const NumericTable * inTable, const NumericTable * outTable, const NumericTable * weights, FPType eps, FPType p,
                       PairwiseDistanceType metric = PairwiseDistanceType::euclidean);

    services::Status queryFull(Neighborhood<FPType, cpu> * neighs, bool doReset = false);

    services::Status query(size_t * indices, size_t n, Neighborhood<FPType, cpu> * neighs, bool doReset = false);
};

template <typename FPType, CpuType cpu>
class NeighborhoodEngine<defaultDense, FPType, cpu>
{
    DAAL_NEW_DELETE();

public:
    NeighborhoodEngine(const NumericTable * inTable, const NumericTable * outTable, const NumericTable * weights, FPType eps, FPType p,
                       PairwiseDistanceType metric = PairwiseDistanceType::euclidean)
        : _inTable(inTable), _outTable(outTable), _weights(weights), _eps(eps), _p(p), _metric(metric)
    {}

    ~NeighborhoodEngine() {}

    NeighborhoodEngine(const NeighborhoodEngine &)             = delete;
    NeighborhoodEngine & operator=(const NeighborhoodEngine &) = delete;

    size_t getIndex(size_t * idx, FPType * array, size_t size, FPType cmp)
    {
        size_t count = 0;

        for (size_t i = 0; i < size; ++i)
        {
            if (array[i] <= cmp)
            {
                idx[count++] = i;
            }
        }
        return count;
    }

    services::Status queryFull(Neighborhood<FPType, cpu> * neighs, bool doReset = false)
    {
        services::SharedPtr<PairwiseDistances<FPType, cpu> > metric(createPairwiseDistances<FPType, cpu>(*_inTable, *_outTable, _metric, (double)_p));
        DAAL_CHECK_MALLOC(metric.get());
        return queryFullImpl(*metric, neighs, doReset);
    }

    services::Status query(size_t * indices, size_t n, Neighborhood<FPType, cpu> * neighs, bool doReset = false)
    {
        // The same engine as `queryFull`, so both paths share `poweredEpsilon`.
        services::SharedPtr<PairwiseDistances<FPType, cpu> > metric(createPairwiseDistances<FPType, cpu>(*_inTable, *_outTable, _metric, (double)_p));
        DAAL_CHECK_MALLOC(metric.get());
        return queryImpl([&](const FPType * a, const FPType * b, size_t dim) { return metric->computeDistance(a, b, dim); }, indices, n, neighs,
                         doReset);
    }

private:
    services::Status queryFullImpl(PairwiseDistances<FPType, cpu> & metric, Neighborhood<FPType, cpu> * neighs, bool doReset)
    {
        SafeStatus safeStat;

        const size_t inRows  = _inTable->getNumberOfRows();
        const size_t outRows = _outTable->getNumberOfRows();

        if (outRows == 0)
        {
            return services::Status();
        }

        const size_t dim    = _inTable->getNumberOfColumns();
        const size_t outDim = _outTable->getNumberOfColumns();
        DAAL_ASSERT(outDim >= dim);

        DAAL_CHECK_STATUS_VAR(metric.init());

        const FPType epsP    = poweredEpsilon<FPType, cpu>(_metric, _eps, _p);
        const bool sameTable = (_inTable == _outTable);

        const size_t inBlockSize = 128;
        const size_t nInBlocks   = inRows / inBlockSize + (inRows % inBlockSize > 0);

        const size_t outBlockSize = 128;
        const size_t nOutBlocks   = outRows / outBlockSize + (outRows % outBlockSize > 0);

        TlsMem<FPType, cpu> tls(inBlockSize * outBlockSize);
        TlsMem<size_t, cpu> tlsIdx(outBlockSize);
        // Hits of an in-block as (row, index) pairs, so that each neighborhood is allocated once per in-block
        // instead of once per out-block.
        daal::tls<Vector<size_t, cpu> *> tlsHits([]() { return new Vector<size_t, cpu>(); });

        TArray<FPType, cpu> onesWeightsArray(outBlockSize);
        FPType * onesWeights = onesWeightsArray.get();
        DAAL_CHECK_MALLOC(onesWeights);
        daal::services::internal::service_memset_seq<FPType, cpu>(onesWeights, FPType(1), outBlockSize);

        daal::threader_for(nInBlocks, 1, [&](size_t inBlock) {
            size_t i1    = inBlock * inBlockSize;
            size_t i2    = (inBlock + 1 == nInBlocks ? inRows : i1 + inBlockSize);
            size_t iSize = i2 - i1;

            ReadRows<FPType, cpu> inDataRows(const_cast<NumericTable *>(_inTable), i1, i2 - i1);
            DAAL_CHECK_BLOCK_STATUS_THR(inDataRows);
            const FPType * const inData = inDataRows.get();

            if (doReset)
            {
                for (size_t i = 0; i < iSize; i++)
                {
                    neighs[i + i1].reset();
                }
            }

            FPType * local = tls.local();

            DAAL_CHECK_MALLOC_THR(local);

            size_t * idx = tlsIdx.local();
            DAAL_CHECK_MALLOC_THR(idx);

            Vector<size_t, cpu> * const hits = tlsHits.local();
            DAAL_CHECK_MALLOC_THR(hits);
            hits->reset();

            for (size_t outBlock = 0; outBlock < nOutBlocks; outBlock++)
            {
                size_t j1    = outBlock * outBlockSize;
                size_t j2    = (outBlock + 1 == nOutBlocks ? outRows : j1 + outBlockSize);
                size_t jSize = j2 - j1;

                ReadRows<FPType, cpu> outDataRows(const_cast<NumericTable *>(_outTable), j1, j2 - j1);
                DAAL_CHECK_BLOCK_STATUS_THR(outDataRows);
                const FPType * const outData = outDataRows.get();

                ReadRows<FPType, cpu> weightsRows;
                if (_weights)
                {
                    weightsRows.set(const_cast<NumericTable *>(_weights), j1, j2 - j1);
                    DAAL_CHECK_BLOCK_STATUS_THR(weightsRows);
                }
                const FPType * const weights = weightsRows.get() ? weightsRows.get() : onesWeights;

                metric.computeBatch(inData, outData, i1, iSize, j1, jSize, local);
                if (sameTable)
                {
                    // A row is always in its own neighborhood.
                    const size_t kStart = services::internal::max<cpu, size_t>(i1, j1);
                    const size_t kEnd   = services::internal::min<cpu, size_t>(i2, j2);
                    for (size_t k = kStart; k < kEnd; k++) local[(k - i1) * jSize + (k - j1)] = FPType(0);
                }

                services::Status status = hits->reserve(2 * iSize * jSize);
                DAAL_CHECK_STATUS_THR(status);
                size_t * pairs = hits->ptr() + hits->size();
                for (size_t i = 0; i < iSize; i++)
                {
                    const size_t indexes = getIndex(idx, local + i * jSize, jSize, epsP);
                    // Per neighbor, in row order: a regrouped float sum can flip core status at the threshold.
                    for (size_t j = 0; j < indexes; j++)
                    {
                        neighs[i + i1].addWeight(weights[idx[j]]);
                        pairs[0] = i;
                        pairs[1] = idx[j] + j1;
                        pairs += 2;
                    }
                }
                hits->resize(pairs - hits->ptr());
            }

            size_t counts[inBlockSize] = { 0 };
            const size_t * const pairs = hits->ptr();
            const size_t pairsEnd      = hits->size();
            for (size_t pos = 0; pos < pairsEnd; pos += 2)
            {
                counts[pairs[pos]]++;
            }
            for (size_t i = 0; i < iSize; i++)
            {
                DAAL_CHECK_MALLOC_THR(!neighs[i + i1].reserve(counts[i]));
            }
            for (size_t pos = 0; pos < pairsEnd; pos += 2)
            {
                neighs[i1 + pairs[pos]].push(pairs[pos + 1]);
            }
        });
        tlsHits.reduce([](Vector<size_t, cpu> * hits) { delete hits; });

        return safeStat.detach();
    }

    template <typename Distance>
    services::Status queryImpl(Distance dist, size_t * indices, size_t n, Neighborhood<FPType, cpu> * neighs, bool doReset)
    {
        SafeStatus safeStat;
        services::Status s;

        const size_t outRows = _outTable->getNumberOfRows();

        if (outRows == 0)
        {
            return s;
        }

        const size_t dim    = _inTable->getNumberOfColumns();
        const size_t outDim = _outTable->getNumberOfColumns();
        DAAL_ASSERT(outDim >= dim);

        services::SharedPtr<NTask<FPType, cpu> > nTask = NTask<FPType, cpu>::create(n);
        DAAL_CHECK_MALLOC(nTask.get());

        TArray<ReadRows<FPType, cpu>, cpu> queryRows(n);
        DAAL_CHECK_MALLOC(queryRows.get());

        for (size_t i = 0; i < n; i++)
        {
            queryRows[i].set(const_cast<NumericTable *>(_inTable), indices[i], 1);
            DAAL_CHECK_BLOCK_STATUS(queryRows[i]);
        }

        if (doReset)
        {
            for (size_t i = 0; i < n; i++)
            {
                neighs[i].reset();
            }
        }

        const FPType epsP    = poweredEpsilon<FPType, cpu>(_metric, _eps, _p);
        const bool sameTable = (_inTable == _outTable);

        size_t outBlockSize = 256;
        size_t nOutBlocks   = outRows / outBlockSize + (outRows % outBlockSize > 0);

        daal::threader_for(nOutBlocks, 1, [&](size_t outBlock) {
            TlsNTask<FPType, cpu> * tlsNTask = nTask->tlsNTask->local();
            DAAL_CHECK_MALLOC_THR(tlsNTask);

            Neighborhood<FPType, cpu> * localNeighs = tlsNTask->neighs;
            DAAL_CHECK_MALLOC_THR(localNeighs);

            size_t j1    = outBlock * outBlockSize;
            size_t j2    = (outBlock + 1 == nOutBlocks ? outRows : j1 + outBlockSize);
            size_t jSize = j2 - j1;

            ReadRows<FPType, cpu> outDataRows(const_cast<NumericTable *>(_outTable), j1, j2 - j1);
            DAAL_CHECK_BLOCK_STATUS_THR(outDataRows);
            const FPType * const outData = outDataRows.get();

            ReadRows<FPType, cpu> weightsRows;
            if (_weights)
            {
                weightsRows.set(const_cast<NumericTable *>(_weights), j1, j2 - j1);
                DAAL_CHECK_BLOCK_STATUS_THR(weightsRows);
            }
            const FPType * const weights = weightsRows.get();

            for (size_t i = 0; i < n; i++)
            {
                for (size_t j = 0; j < jSize; j++)
                {
                    // A row is always in its own neighborhood, as in `queryFull`.
                    if ((sameTable && indices[i] == j + j1) || dist(queryRows[i].get(), &outData[j * outDim], dim) <= epsP)
                    {
                        DAAL_CHECK_MALLOC_THR(!localNeighs[i].add(j + j1, (weights ? weights[j] : (FPType)1.0)));
                    }
                }
            }
        });

        s = safeStat.detach();
        DAAL_CHECK_STATUS_VAR(s);

        nTask->tlsNTask->reduce([&](TlsNTask<FPType, cpu> * tt) -> void {
            Neighborhood<FPType, cpu> * localNeighs = tt->neighs;
            if (localNeighs)
            {
                for (size_t i = 0; i < n; i++)
                {
                    size_t localSize = localNeighs[i].size();
                    for (size_t j = 0; j < localSize; j++)
                    {
                        neighs[i].add(localNeighs[i].get(j), 0);
                    }
                    neighs[i].addWeight(localNeighs[i].weight());
                }
            }
        });

        return s;
    }

    const NumericTable * _inTable;
    const NumericTable * _outTable;
    const NumericTable * _weights;

    FPType _eps;
    FPType _p;
    PairwiseDistanceType _metric;
};

template <typename FPType, CpuType cpu>
FPType findKthStatistic(FPType * values, size_t nElements, size_t k)
{
    if (nElements == 0)
    {
        return (FPType)0;
    }

    if (k >= nElements)
    {
        k = nElements - 1;
    }

    size_t l = 0;
    size_t r = nElements - 1;
    while (l < r)
    {
        FPType med = values[k];
        int i      = l;
        int j      = r;
        while (i <= j)
        {
            while (values[i] < med)
            {
                i++;
            }
            while (med < values[j])
            {
                j--;
            }
            if (i <= j)
            {
                daal::services::internal::swap<cpu, FPType>(values[i], values[j]);
                i++;
                j--;
            }
        }
        if (j < k)
        {
            l = i;
        }
        if (k < i)
        {
            r = j;
        }
    }

    return values[k];
}

} // namespace internal
} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
