/* file: dtrees_feature_type_helper.i */
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
//  Cpu-dependent initialization of service data structure
//--
*/

#include "include/services/error_indexes.h"
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_sort.h"
#include "src/algorithms/dtrees/service_array.h"
#include "src/externals/service_memory.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{
template <typename IndexType, typename algorithmFPType, CpuType cpu>
struct ColIndexTask
{
    DAAL_NEW_DELETE();
    ColIndexTask(size_t nRows) : _index(nRows), maxNumDiffValues(1) {}
    virtual ~ColIndexTask() {}
    bool isValid() const { return _index.get(); }

    struct FeatureIdx
    {
        algorithmFPType key;
        IndexType val;
        static int compare(const void * a, const void * b)
        {
            if (static_cast<const FeatureIdx *>(a)->key < static_cast<const FeatureIdx *>(b)->key) return -1;
            return static_cast<const FeatureIdx *>(a)->key > static_cast<const FeatureIdx *>(b)->key;
        }
        bool operator<(const FeatureIdx & o) const { return key < o.key; }
    };

    virtual services::Status makeIndex(NumericTable & nt, IndexedFeatures::FeatureEntry & entry, IndexType * aRes, size_t iCol, size_t nRows,
                                       bool bUnorderedFeature)
    {
        return this->template makeIndexDefault<false>(nt, entry, aRes, iCol, nRows, bUnorderedFeature);
    }

    template <bool binLabels>
    services::Status makeIndexDefault(NumericTable & nt, IndexedFeatures::FeatureEntry & entry, IndexType * aRes, size_t iCol, size_t nRows,
                                      bool bUnorderedFeature)
    {
        Status s = this->getSorted(nt, iCol, nRows);
        if (!s) return s;
        const FeatureIdx * index = _index.get();
        if (index[0].key == index[nRows - 1].key)
        {
            entry.numIndices = 1;
            for (size_t i = 0; i < nRows; ++i) aRes[i] = 0;

            if (binLabels)
            {
                s |= entry.allocBorders();
                DAAL_CHECK(s, s);
                entry.min           = index[0].key;
                entry.binBorders[0] = index[nRows - 1].key;
            }

            return s;
        }
        IndexType iUnique    = 0;
        aRes[index[0].val]   = iUnique;
        algorithmFPType prev = index[0].key;
        for (size_t i = 1; i < nRows; ++i)
        {
            const IndexType idx = index[i].val;
            if (index[i].key == prev)
                aRes[idx] = iUnique;
            else
            {
                aRes[idx] = ++iUnique;
                prev      = index[i].key;
            }
        }
        ++iUnique;
        entry.numIndices = iUnique;
        if (maxNumDiffValues < iUnique) maxNumDiffValues = iUnique;

        if (binLabels)
        {
            s |= entry.allocBorders();

            IndexType iUnique    = 0;
            algorithmFPType prev = index[0].key;
            entry.min            = prev;
            entry.binBorders[0]  = prev;

            for (size_t i = 1; i < nRows; ++i)
            {
                if (index[i].key != prev)
                {
                    prev                        = index[i].key;
                    entry.binBorders[++iUnique] = prev;
                }
            }
        }

        return services::Status();
    }

public:
    size_t maxNumDiffValues;

protected:
    Status getSorted(NumericTable & nt, size_t iCol, size_t nRows)
    {
        const algorithmFPType * pBlock = _block.set(&nt, iCol, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(_block);
        FeatureIdx * index = _index.get();
        for (size_t i = 0; i < nRows; ++i)
        {
            index[i].key = pBlock[i];
            index[i].val = i;
        }
        daal::algorithms::internal::qSortByKey<FeatureIdx, cpu>(nRows, index);
        return Status();
    }

protected:
    daal::internal::ReadColumns<algorithmFPType, cpu> _block;
    TVector<FeatureIdx, cpu, DefaultAllocator<cpu> > _index;
};

template <typename IndexType, typename algorithmFPType, CpuType cpu>
struct ColIndexTaskBins : public ColIndexTask<IndexType, algorithmFPType, cpu>
{
    typedef ColIndexTask<IndexType, algorithmFPType, cpu> super;
    ColIndexTaskBins(size_t nRows, const BinParams & prm) : super(nRows), _prm(prm), _bins(_prm.maxBins) {}

    /*
    * Transform features based on the BinParams _prm.
    *   - If no BinParams _prm are provided, one bin per unique value in the
    *     dataset is created
    *   - If BinParams _prm are provided, the strategy set according to
    *     BinParams::Strategy is used
    */
    virtual services::Status makeIndex(NumericTable & nt, IndexedFeatures::FeatureEntry & entry, IndexType * aRes, size_t iCol, size_t nRows,
                                       bool bUnorderedFeature) override;
    /* Function to create feature indices for Strategy == quantiles */
    services::Status makeIndexQuantiles(NumericTable & nt, IndexedFeatures::FeatureEntry & entry, IndexType * aRes, size_t iCol, size_t nRows);
    /* Function to create feature indices for Strategy == averages */
    services::Status makeIndexAverages(NumericTable & nt, IndexedFeatures::FeatureEntry & entry, IndexType * aRes, size_t iCol, size_t nRows);
    /* Helper to treat constant-valued features */
    services::Status makeIndexConstant(IndexedFeatures::FeatureEntry & entry, IndexType * aRes, size_t nRows);

private:
    services::Status assignIndexAccordingToBins(IndexedFeatures::FeatureEntry & entry, IndexType * aRes, size_t nBins, size_t nRows);

private:
    const BinParams _prm;
    TVector<size_t, cpu, DefaultAllocator<cpu> > _bins;
};

template <typename TContainer>
static void append(TContainer & cont, size_t & contSize, size_t size)
{
    cont[contSize++] = size;
}

//Returns an index of the first element in the range[ar, ar + n) that is not less than(i.e.greater or equal to) value.
template <typename T>
const T * upper_bound(const T * first, const T * last, const T & value)
{
    size_t n = last - first;
    while (n > 0)
    {
        auto it   = first;
        auto step = (n >> 1);
        it += step;
        if (!(value < *it))
        {
            first = ++it;
            n -= step + 1;
        }
        else
            n = step;
    }
    return first;
}

template <typename IndexType, typename algorithmFPType, CpuType cpu>
services::Status ColIndexTaskBins<IndexType, algorithmFPType, cpu>::assignIndexAccordingToBins(IndexedFeatures::FeatureEntry & entry,
                                                                                               IndexType * aRes, size_t nBins, size_t nRows)
{
    const typename super::FeatureIdx * index = this->_index.get();

    entry.min = index[0].key;

    if (nBins == 1)
    {
        entry.numIndices   = 1;
        services::Status s = entry.allocBorders();
        DAAL_CHECK(s, s);
        services::internal::service_memset_seq<IndexType, cpu>(aRes, 0, nRows);

        entry.binBorders[0] = index[nRows - 1].key;
        _bins[0]            = nRows;
        return Status();
    }
    entry.numIndices   = nBins;
    services::Status s = entry.allocBorders();
    if (!s) return s;

    size_t i = 0;
    for (size_t iBin = 0; iBin < nBins; ++iBin)
    {
        for (size_t n = i + _bins[iBin]; i < n; ++i) aRes[index[i].val] = iBin;
        entry.binBorders[iBin] = index[i - 1].key;
    }
    if (this->maxNumDiffValues < entry.numIndices) this->maxNumDiffValues = entry.numIndices;
    return s;
}

template <typename IndexType, typename algorithmFPType, CpuType cpu>
services::Status ColIndexTaskBins<IndexType, algorithmFPType, cpu>::makeIndex(NumericTable & nt, IndexedFeatures::FeatureEntry & entry,
                                                                              IndexType * aRes, size_t iCol, size_t nRows, bool bUnorderedFeature)
{
    /* feature is not ordered or fewer data points than bins -> no indexing needed */
    if (bUnorderedFeature || nRows <= _prm.maxBins) return this->template makeIndexDefault<true>(nt, entry, aRes, iCol, nRows, bUnorderedFeature);

    /* sort feature values */
    Status s = this->getSorted(nt, iCol, nRows);
    if (!s) return s;

    /* special case: all values are the same -> constant-valued feature */
    const typename super::FeatureIdx * index = this->_index.get();
    if (index[0].key == index[nRows - 1].key)
    {
        return makeIndexConstant(entry, aRes, nRows);
    }

    /* Create bins of sorted data according to strategy selected in _prm */
    switch (_prm.binningStrategy)
    {
    case dtrees::internal::BinningStrategy::quantiles: return makeIndexQuantiles(nt, entry, aRes, iCol, nRows);
    case dtrees::internal::BinningStrategy::averages: return makeIndexAverages(nt, entry, aRes, iCol, nRows);
    default: return Status(ErrorID::ErrorMethodNotSupported);
    }
}

template <typename IndexType, typename algorithmFPType, CpuType cpu>
services::Status ColIndexTaskBins<IndexType, algorithmFPType, cpu>::makeIndexQuantiles(NumericTable & nt, IndexedFeatures::FeatureEntry & entry,
                                                                                       IndexType * aRes, size_t iCol, size_t nRows)
{
    const typename super::FeatureIdx * index = this->_index.get();

    size_t nBins = 0;
    DAAL_ASSERT(_prm.maxBins > 0);
    const size_t binSize = nRows / _prm.maxBins;
    int64_t remainder    = nRows % _prm.maxBins; //allow for negative values
    size_t dx            = 2 * _prm.maxBins;
    size_t dy            = 2 * remainder;
    int64_t D            = dy - _prm.maxBins; //use bresenham's line algorithm to distribute remainder

    size_t i = 0;
    for (; (i + binSize + 1 < nRows) && (nBins < _prm.maxBins);)
    {
        //trying to make a bin of size binSize
        size_t newBinSize = binSize;
        if (remainder > 0)
        {
            if (D > 0)
            {
                newBinSize++;
                remainder--;
                D -= dx;
            }
            D += dy;
        }
        size_t iRight                         = i + newBinSize - 1; //intersperse remainder amongst bins
        const typename super::FeatureIdx & ri = index[iRight];

        if (ri.key != index[iRight + 1].key)
        {
            // value changed from one bin to the next, append and continue
            append(_bins, nBins, newBinSize);
            i += newBinSize;
            continue;
        }

        /* when arriving here, the feature value has not changed and
         * we have to move iRight to the right until we find a new value
         * r will be located at the first value that is different from ri.key
         */
        ++iRight;
        size_t r = iRight + binSize;
        while (r < nRows && index[r].key == ri.key)
        {
            r += binSize;
        }
        if (r > nRows)
        {
            r = nRows;
        }
        // upper_bound() returns the index of the first value change between
        // index + iRight + 1 and index + r
        iRight     = upper_bound<typename super::FeatureIdx>(index + iRight + 1, index + r, ri) - index;
        newBinSize = iRight - i;

        if (newBinSize >= 2 * binSize)
        {
            // the new bin is too wide, try insert an additional bin to the left
            size_t iClosestSmallerValue = i + binSize - 1;
            while (iClosestSmallerValue > i && index[iClosestSmallerValue].key == ri.key)
            {
                --iClosestSmallerValue;
            }
            size_t dist = iClosestSmallerValue - i;
            if (dist > _prm.minBinSize)
            {
                // add an extra bin at the left
                const size_t newLeftBinSize = dist + 1;
                append(_bins, nBins, newLeftBinSize);
                i += newLeftBinSize;
                newBinSize -= newLeftBinSize;
            }
            else if ((nBins > 0) && dist > 0)
            {
                // no room for an extra bin to the left, extend the previous
                // one if possible
                const size_t nAddToPrevBin = dist + 1;
                _bins[nBins - 1] += nAddToPrevBin;
                i += nAddToPrevBin;
                newBinSize -= nAddToPrevBin;
            }
            if (remainder > 0)
            { //reset bresenhams line due to unexpected change in remainder
                remainder -= newBinSize - binSize;
                dx = 2 * (_prm.maxBins - nBins - 1);
                dy = 2 * remainder;
                D  = dy - _prm.maxBins + nBins + 1;
            }
        }

        // append the bin and continue
        append(_bins, nBins, newBinSize);
        i += newBinSize;
    }

    // collect the remaining data rows in the final bin
    if (i < nRows)
    {
        size_t newBinSize = nRows - i;
        if (((nBins < _prm.maxBins) && (newBinSize >= _prm.minBinSize)) || nBins == 0)
        {
            append(_bins, nBins, newBinSize);
        }
        else
        {
            _bins[nBins - 1] += newBinSize;
        }
    }

#if _DEBUG
    #if 0
    //run-time check for bins correctness
    size_t nTotal = 0;
    for(size_t i = 0; i < nBins; nTotal += _bins[i], ++i);
    DAAL_ASSERT(nTotal == nRows);
    size_t iBorder = 0;
    for(size_t i = 1; i < nBins; ++i)
    {
        iBorder += _bins[i - 1];
        DAAL_ASSERT(index[iBorder - 1].key != index[iBorder].key);
    }
    #endif
#endif
    return assignIndexAccordingToBins(entry, aRes, nBins, nRows);
}

template <typename IndexType, typename algorithmFPType, CpuType cpu>
services::Status ColIndexTaskBins<IndexType, algorithmFPType, cpu>::makeIndexAverages(NumericTable & nt, IndexedFeatures::FeatureEntry & entry,
                                                                                      IndexType * aRes, size_t iCol, size_t nRows)
{
    const typename super::FeatureIdx * index = this->_index.get();

    size_t nBins = 0;
    size_t i     = 0;
    DAAL_ASSERT(_prm.maxBins > 0);
    algorithmFPType binSize = (index[nRows - 1].key - index[0].key) / _prm.maxBins;
    algorithmFPType value   = index[0].key;

    while (i < nRows)
    {
        // next bin border to the right of current index
        size_t iRight = i + 1;

        while ((iRight < nRows) && (index[iRight].key < (value + binSize)))
        {
            ++iRight;
        }

        // found a new binEdge
        // append the bin and continue
        size_t newBinSize = iRight - i;

        append(_bins, nBins, newBinSize);

        i     = iRight;
        value = index[i].key;
    }

    // assert we picked up all data records
    DAAL_ASSERT(i == nRows);
    DAAL_ASSERT(nBins <= _prm.maxBins);

    return assignIndexAccordingToBins(entry, aRes, nBins, nRows);
}

template <typename IndexType, typename algorithmFPType, CpuType cpu>
services::Status ColIndexTaskBins<IndexType, algorithmFPType, cpu>::makeIndexConstant(IndexedFeatures::FeatureEntry & entry, IndexType * aRes,
                                                                                      size_t nRows)
{
    const typename super::FeatureIdx * index = this->_index.get();

    _bins[0] = nRows;
    services::internal::service_memset_seq<IndexType, cpu>(aRes, 0, nRows);

    entry.numIndices = 1;
    Status s         = entry.allocBorders();
    DAAL_CHECK(s, s);
    entry.binBorders[0] = index[nRows - 1].key;
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status IndexedFeatures::init(const NumericTable & nt, const FeatureTypes * featureTypes, const BinParams * pBimPrm)
{
    dtrees::internal::FeatureTypes autoFT;
    if (!featureTypes)
    {
        DAAL_CHECK_MALLOC(autoFT.init(nt));
        featureTypes = &autoFT;
    }

    _maxNumIndices     = 0;
    services::Status s = alloc(nt.getNumberOfColumns(), nt.getNumberOfRows());
    if (!s) return s;

    const size_t nC = nt.getNumberOfColumns();
    typedef ColIndexTask<IndexType, algorithmFPType, cpu> TlsTask;
    typedef ColIndexTask<IndexType, algorithmFPType, cpu> DefaultTask;
    typedef ColIndexTaskBins<IndexType, algorithmFPType, cpu> BinningTask;

    daal::tls<TlsTask *> tlsData([=, &nt]() -> TlsTask * {
        const size_t nRows = nt.getNumberOfRows();
        TlsTask * res      = (!pBimPrm || (pBimPrm->maxBins == 0)) ? new DefaultTask(nRows) : new BinningTask(nRows, *pBimPrm);
        if (res && !res->isValid())
        {
            delete res;
            res = nullptr;
        }
        return res;
    });

    SafeStatus safeStat;
    daal::threader_for(nC, nC, [&](size_t iCol) {
        //in case of single thread no need to allocate
        TlsTask * task = tlsData.local();
        DAAL_CHECK_THR(task, services::ErrorMemoryAllocationFailed);
        safeStat |=
            task->makeIndex(const_cast<NumericTable &>(nt), _entries[iCol], _data + iCol * nRows(), iCol, nRows(), featureTypes->isUnordered(iCol));
    });
    tlsData.reduce([&](TlsTask * task) -> void {
        if (_maxNumIndices < task->maxNumDiffValues) _maxNumIndices = task->maxNumDiffValues;
        delete task;
    });
    return safeStat.detach();
}

} /* namespace internal */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */
