/* file: dtrees_feature_type_helper.h */
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
//  Implementation of a service class that provides optimal access to the feature types
//--
*/

#ifndef __DTREES_FEATURE_TYPE_HELPER_H__
#define __DTREES_FEATURE_TYPE_HELPER_H__

#include "include/algorithms/decision_forest/decision_forest_training_parameter.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"

typedef double ModelFPType;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// Helper class, provides optimal access to the feature types
//////////////////////////////////////////////////////////////////////////////////////////
class FeatureTypes
{
public:
    FeatureTypes() : _bAllUnordered(false) {}
    ~FeatureTypes();
    bool init(const NumericTable & data);

    bool isUnordered(size_t iFeature) const { return _bAllUnordered || (_aFeat && findInBuf(iFeature)); }

    bool hasUnorderedFeatures() const { return (_bAllUnordered || _nNoOrderedFeat); }

    size_t getNumberOfFeatures() const { return _nFeat; }

    void clearBuf() { destroyBuf(); }

private:
    void allocBuf(size_t n);
    void destroyBuf();
    bool findInBuf(size_t iFeature) const;

private:
    bool * _aFeat          = nullptr; //buffer with minimal required features data
    size_t _nFeat          = 0;       //size of the buffer
    size_t _nNoOrderedFeat = 0;
    bool _bAllUnordered    = false;
    int _firstUnordered    = -1;
    int _lastUnordered     = -1;
};

using daal::algorithms::decision_forest::training::BinningStrategy;

struct BinParams
{
    BinParams(size_t _maxBins, size_t _minBinSize, BinningStrategy _binningStrategy = BinningStrategy::quantiles)
        : maxBins(_maxBins), minBinSize(_minBinSize), binningStrategy(_binningStrategy)
    {}
    BinParams(const BinParams & o) : maxBins(o.maxBins), minBinSize(o.minBinSize), binningStrategy(o.binningStrategy) {}

    /* Maximum number of bins for indexed data. Default: 256 */
    size_t maxBins = 256;
    /* Minimum bin width (number of data points per bin). Default: 5*/
    size_t minBinSize = 5;
    /* Strategy to create bins for feature values. Default: quantiles */
    BinningStrategy binningStrategy = BinningStrategy::quantiles;
};

//////////////////////////////////////////////////////////////////////////////////////////
// IndexedFeatures. Creates and stores index of every feature
// Sorts every feature and creates the mapping: features value -> index of the value
// in the sorted array of unique values of the feature in increasing order
//////////////////////////////////////////////////////////////////////////////////////////
class IndexedFeatures
{
public:
    typedef int IndexType; // TODO: should be unsigned int

    struct FeatureEntry
    {
        DAAL_NEW_DELETE();
        IndexType numIndices     = 0;       // number of indices or bins
        ModelFPType * binBorders = nullptr; // right bin borders
        ModelFPType min          = 0;       // used for random splitter, since all borders are known but min.

        services::Status allocBorders();
        ~FeatureEntry();
    };

public:
    IndexedFeatures() : _data(nullptr), _entries(nullptr), _sizeOfIndex(sizeof(IndexType)), _nRows(0), _nCols(0), _capacity(0), _maxNumIndices(0) {}

    ~IndexedFeatures();

    //initialize the feature indices, i.e. bins
    template <typename algorithmFPType, CpuType cpu>
    services::Status init(const NumericTable & nt, const FeatureTypes * featureTypes = nullptr, const BinParams * pBimPrm = nullptr);

    //get max number of indices for that feature
    IndexType numIndices(size_t iCol) const { return _entries[iCol].numIndices; }

    //get max number of indices among all features
    IndexType maxNumIndices() const { return _maxNumIndices; }

    //returns true if the feature is mapped to bins
    bool isBinned(size_t iCol) const
    {
        DAAL_ASSERT(iCol < _nCols);
        return !!_entries[iCol].binBorders;
    }

    //returns right border of the bin if the feature is a binned one
    ModelFPType binRightBorder(size_t iCol, size_t iBin) const
    {
        DAAL_ASSERT(isBinned(iCol));
        DAAL_ASSERT(iBin < static_cast<size_t>(numIndices(iCol)));
        return _entries[iCol].binBorders[iBin];
    }

    //returns right border of the bin if the feature is a binned one
    ModelFPType min(size_t iCol) const
    {
        DAAL_ASSERT(isBinned(iCol));
        return _entries[iCol].min;
    }

    //for low-level optimization
    const IndexType * data(size_t iFeature) const { return (IndexType *)(((char *)_data) + _nRows * iFeature * _sizeOfIndex); }

    size_t nRows() const { return _nRows; }
    size_t nCols() const { return _nCols; }

protected:
    services::Status alloc(size_t nCols, size_t nRows);

protected:
    IndexType * _data;
    FeatureEntry * _entries;
    size_t _sizeOfIndex;
    size_t _nRows;
    size_t _nCols;
    size_t _capacity;
    size_t _maxNumIndices;
};

} /* namespace internal */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */

#endif
