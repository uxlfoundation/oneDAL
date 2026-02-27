/** file soa_numeric_table.cpp */
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

#include "data_management/data/soa_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
SOANumericTable::SOANumericTable(NumericTableDictionary * ddict, size_t nRows, AllocationFlag memoryAllocationFlag)
    : NumericTable(NumericTableDictionaryPtr(ddict, services::EmptyDeleter())), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    _index  = 0;

    this->_status |= setNumberOfRowsImpl(nRows);
    if (!resizePointersArray(getNumberOfColumns()))
    {
        this->_status.add(services::ErrorMemoryAllocationFailed);
        return;
    }
    if (memoryAllocationFlag == doAllocate)
    {
        this->_status |= allocateDataMemoryImpl();
    }
}

SOANumericTable::SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual)
    : NumericTable(nColumns, nRows, featuresEqual), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    _index  = 0;

    if (!resizePointersArray(nColumns))
    {
        this->_status.add(services::ErrorMemoryAllocationFailed);
        return;
    }
}

services::SharedPtr<SOANumericTable> SOANumericTable::create(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual,
                                                             services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(SOANumericTable, nColumns, nRows, featuresEqual);
}

SOANumericTable::SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag)
    : NumericTable(ddict), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    _index  = 0;

    this->_status |= setNumberOfRowsImpl(nRows);

    if (!resizePointersArray(getNumberOfColumns()))
    {
        this->_status.add(services::ErrorMemoryAllocationFailed);
        return;
    }
    if (memoryAllocationFlag == doAllocate)
    {
        this->_status |= allocateDataMemoryImpl();
    }
}

services::SharedPtr<SOANumericTable> SOANumericTable::create(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag,
                                                             services::Status * stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(SOANumericTable, ddict, nRows, memoryAllocationFlag);
}

SOANumericTable::SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status & st)
    : NumericTable(nColumns, nRows, featuresEqual, st), _arrays(nColumns), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    _index  = 0;
    if (!resizePointersArray(nColumns))
    {
        st.add(services::ErrorMemoryAllocationFailed);
        return;
    }
}

SOANumericTable::SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status & st)
    : NumericTable(ddict, st), _arraysInitialized(0), _partialMemStatus(notAllocated)
{
    _layout = soa;
    _index  = 0;
    st |= setNumberOfRowsImpl(nRows);
    if (!resizePointersArray(getNumberOfColumns()))
    {
        st.add(services::ErrorMemoryAllocationFailed);
        return;
    }
    if (memoryAllocationFlag == doAllocate)
    {
        st |= allocateDataMemoryImpl();
    }
}

bool SOANumericTable::isHomogeneousFloatOrDouble() const
{
    const size_t ncols                                      = getNumberOfColumns();
    const NumericTableFeature & f0                          = (*_ddict)[0];
    daal::data_management::features::IndexNumType indexType = f0.indexType;

    for (size_t i = 1; i < ncols; ++i)
    {
        const NumericTableFeature & f1 = (*_ddict)[i];
        if (f1.indexType != indexType) return false;
    }

    return indexType == daal::data_management::features::getIndexNumType<float>()
           || indexType == daal::data_management::features::getIndexNumType<double>();
}

bool SOANumericTable::isAllCompleted() const
{
    return _arraysInitialized == getNumberOfColumns();
}

services::Status SOANumericTable::searchMinPointer()
{
    const size_t ncols = getNumberOfColumns();

    DAAL_CHECK_MALLOC(_wrapOffsets.allocate(ncols));
    _index              = 0;
    char const * ptrMin = (char *)_arrays[0].get();

    /* search index for min pointer */
    for (size_t i = 1; i < ncols; ++i)
    {
        if ((char *)_arrays[i].get() < ptrMin)
        {
            _index = i;
            ptrMin = (char *)_arrays[i].get();
        }
    }

    DAAL_ASSERT(_wrapOffsets.count() >= ncols)

    /* compute offsets */
    for (size_t i = 0; i < ncols; ++i)
    {
        char const * const pv = (char *)(_arrays[i].get());
        /* unsigned long long is equal to DAAL_UINT64 and LLONG_MAX is always fit to unsigned long long */
        DAAL_ASSERT(static_cast<DAAL_UINT64>(pv - ptrMin) <= static_cast<DAAL_UINT64>(LLONG_MAX))
        _wrapOffsets.get()[i] = static_cast<DAAL_INT64>(pv - ptrMin);
    }

    return services::Status();
}

bool SOANumericTable::resizePointersArray(size_t nColumns)
{
    if (_arrays.size() >= nColumns)
    {
        size_t counter = 0;
        for (size_t i = 0; i < nColumns; i++)
        {
            counter += (_arrays[i] != 0);
        }
        _arraysInitialized = counter;

        if (_arraysInitialized == nColumns)
        {
            _memStatus = _partialMemStatus;
        }
        else
        {
            _memStatus = notAllocated;
        }

        return true;
    }

    bool is_resized = _arrays.resize(nColumns);
    if (is_resized)
    {
        _memStatus = notAllocated;
    }

    _wrapOffsets.deallocate();
    _index = 0;

    return is_resized;
}

services::Status SOANumericTable::setNumberOfColumnsImpl(size_t ncol)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, NumericTable::setNumberOfColumnsImpl(ncol));

    if (!resizePointersArray(ncol))
    {
        return services::Status(services::ErrorMemoryAllocationFailed);
    }
    return s;
}

void SOANumericTable::freeDataMemoryImpl()
{
    _arrays.clear();
    _arrays.resize(_ddict->getNumberOfFeatures());
    _arraysInitialized = 0;

    _partialMemStatus = notAllocated;
    _memStatus        = notAllocated;
}

services::Status SOANumericTable::allocateDataMemoryImpl(daal::MemType /*type*/)
{
    freeDataMemoryImpl();

    size_t ncol  = _ddict->getNumberOfFeatures();
    size_t nrows = getNumberOfRows();

    if (ncol * nrows == 0)
    {
        if (nrows == 0)
        {
            return services::Status(services::ErrorIncorrectNumberOfObservations);
        }
        else
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }
    }

    for (size_t i = 0; i < ncol; i++)
    {
        NumericTableFeature f = (*_ddict)[i];
        if (f.typeSize != 0)
        {
            _arrays[i] = services::SharedPtr<byte>((byte *)daal::services::daal_malloc(f.typeSize * nrows), services::ServiceDeleter());
            _arraysInitialized++;
        }
        if (!_arrays[i])
        {
            freeDataMemoryImpl();
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
    }

    if (_arraysInitialized > 0)
    {
        _partialMemStatus = internallyAllocated;
    }

    if (_arraysInitialized == ncol)
    {
        _memStatus = internallyAllocated;
    }

    DAAL_CHECK_STATUS_VAR(generatesOffsets());

    return services::Status();
}

} // namespace interface1
} // namespace data_management
} // namespace daal
