/* file: dtrees_feature_type_helper.cpp */
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
//  Implementation of service data structures
//--
*/
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{
FeatureTypes::~FeatureTypes()
{
    destroyBuf();
}

bool FeatureTypes::init(const NumericTable & data)
{
    size_t count    = 0;
    _firstUnordered = -1;
    _lastUnordered  = -1;
    _nFeat          = data.getNumberOfColumns();

    // All features have the type of the first one in a dictionary with equal features
    const data_management::NumericTableDictionaryPtr dict = data.getDictionarySharedPtr();
    if (_nFeat && dict && dict->getFeaturesEqual() == data_management::DictionaryIface::equal
        && data.getFeatureType(0) != data_management::features::DAAL_CATEGORICAL)
    {
        _bAllUnordered = false;
        return true;
    }

    for (size_t i = 0; i < _nFeat; ++i)
    {
        if (data.getFeatureType(i) != data_management::features::DAAL_CATEGORICAL) continue;
        if (_firstUnordered < 0) _firstUnordered = i;
        _lastUnordered = i;
        ++count;
    }
    _bAllUnordered = ((_nNoOrderedFeat == count) && count);
    if (_bAllUnordered)
    {
        destroyBuf();
        return true;
    }
    if (!count) return true;
    allocBuf(_lastUnordered - _firstUnordered + 1);
    if (!_aFeat) return false;
    for (size_t i = _firstUnordered; i < _lastUnordered + 1; ++i)
    {
        _aFeat[i - _firstUnordered] = (data.getFeatureType(i) == data_management::features::DAAL_CATEGORICAL);
    }
    return true;
}

void FeatureTypes::allocBuf(size_t n)
{
    destroyBuf();
    if (n)
    {
        _nNoOrderedFeat = n;
        _aFeat          = (bool *)daal::services::daal_calloc(_nNoOrderedFeat);
        if (!_aFeat) _nNoOrderedFeat = 0;
    }
}

void FeatureTypes::destroyBuf()
{
    if (_aFeat)
    {
        daal::services::daal_free(_aFeat);
        _aFeat          = nullptr;
        _nNoOrderedFeat = 0;
    }
}

bool FeatureTypes::findInBuf(size_t iFeature) const
{
    if (iFeature < _firstUnordered) return false;
    const size_t i = iFeature - _firstUnordered;
    if (i < _nNoOrderedFeat) return _aFeat[i];
    DAAL_ASSERT(iFeature > _lastUnordered);
    return false;
}

IndexedFeatures::~IndexedFeatures()
{
    if (_data) daal::services::daal_free(_data);
    delete[] _entries;
    _data    = nullptr;
    _entries = nullptr;
}

IndexedFeatures::FeatureEntry::~FeatureEntry()
{
    if (binBorders) daal::services::daal_free(binBorders);
    binBorders = nullptr;
}

services::Status IndexedFeatures::FeatureEntry::allocBorders()
{
    if (binBorders)
    {
        daal::services::daal_free(binBorders);
        binBorders = nullptr;
    }
    binBorders = (ModelFPType *)services::daal_calloc(sizeof(ModelFPType) * numIndices);
    return binBorders ? services::Status() : services::Status(services::ErrorMemoryAllocationFailed);
}

services::Status IndexedFeatures::alloc(size_t nC, size_t nR)
{
    const size_t newCapacity = nC * nR;
    if (_data)
    {
        if (newCapacity > _capacity)
        {
            services::daal_free(_data);
            _data     = nullptr;
            _capacity = 0;
            _data     = (IndexType *)services::daal_calloc(sizeof(IndexType) * newCapacity);
            DAAL_CHECK_MALLOC(_data);
            _capacity = newCapacity;
        }
    }
    else
    {
        _data = (IndexType *)services::daal_calloc(sizeof(IndexType) * newCapacity);
        DAAL_CHECK_MALLOC(_data);
        _capacity = newCapacity;
    }
    if (_entries)
    {
        delete[] _entries;
        _entries = nullptr;
    }
    _entries = new FeatureEntry[nC];
    DAAL_CHECK_MALLOC(_entries);
    _nCols = nC;
    _nRows = nR;
    return services::Status();
}

} /* namespace internal */
} /* namespace dtrees */
} /* namespace algorithms */
} /* namespace daal */
