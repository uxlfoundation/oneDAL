/* file: csv_data_source.h */
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
//  Implementation of the file data source class.
//--
*/

#ifndef __CSV_DATA_SOURCE_H__
#define __CSV_DATA_SOURCE_H__

#include "services/daal_memory.h"

#include "data_management/data_source/data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data_source/internal/data_source_options.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup data_sources
 * @{
 */

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CSVDATASOURCEOPTIONS"></a>
 *  \brief Options of CSV data source
 */
class CsvDataSourceOptions
{
public:
    enum Value
    {
        byDefault                   = 0,
        allocateNumericTable        = 1 << 0,
        createDictionaryFromContext = 1 << 1,
        parseHeader                 = 1 << 2
    };

    static CsvDataSourceOptions::Value unite(const CsvDataSourceOptions::Value & lhs, const CsvDataSourceOptions::Value & rhs)
    {
        return internal::DataSourceOptionsImpl<Value>::unite(lhs, rhs);
    }

    CsvDataSourceOptions(Value flags = byDefault) : _impl(flags) {}

    DataSource::NumericTableAllocationFlag getNumericTableAllocationFlag() const
    {
        return (_impl.getFlag(allocateNumericTable)) ? DataSource::doAllocateNumericTable : DataSource::notAllocateNumericTable;
    }

    DataSource::DictionaryCreationFlag getDictionaryCreationFlag() const
    {
        return (_impl.getFlag(createDictionaryFromContext)) ? DataSource::doDictionaryFromContext : DataSource::notDictionaryFromContext;
    }

    bool getParseHeaderFlag() const { return _impl.getFlag(parseHeader); }

private:
    internal::DataSourceOptionsImpl<Value> _impl;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__FILEDATASOURCE"></a>
 *  \brief Specifies methods to access data stored in files
 *  \tparam FeatureManager         The type of feature manager that specifies how to extract numerical data from CSV
 *  \tparam SummaryStatisticsType  The floating point type to compute summary statics for numeric table
 */
template <typename FeatureManager, typename SummaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE>
class CsvDataSource : public DataSourceTemplate<data_management::HomogenNumericTable<DAAL_DATA_TYPE>, SummaryStatisticsType>
{
private:
    typedef data_management::HomogenNumericTable<DAAL_DATA_TYPE> DefaultNumericTableType;
    typedef DataSourceTemplate<DefaultNumericTableType, SummaryStatisticsType> super;

protected:
    using super::_dict;
    using super::_initialMaxRows;

public:
    /**
     *  Main constructor for a Data Source
     *  \param[in]  doAllocateNumericTable          Flag that specifies whether a Numeric Table
     *                                              associated with a File Data Source is allocated inside the Data Source
     *  \param[in]  doCreateDictionaryFromContext   Flag that specifies whether a Data %Dictionary
     *                                              is created from the context of the File Data Source
     *  \param[in]  initialMaxRows                  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    CsvDataSource(DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable    = DataSource::notAllocateNumericTable,
                  DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                  size_t initialMaxRows                                                 = 10)
        : super(doAllocateNumericTable, doCreateDictionaryFromContext)
    {
        initialize(initialMaxRows);
    }

    /**
     *  Main constructor for a Data Source
     *  \param[in]  options         Options of data source
     *  \param[in]  initialMaxRows  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    CsvDataSource(const CsvDataSourceOptions & options, size_t initialMaxRows = 10)
        : super(options.getNumericTableAllocationFlag(), options.getDictionaryCreationFlag())
    {
        initialize(initialMaxRows);
        _parseHeader = options.getParseHeaderFlag();
    }

    virtual ~CsvDataSource()
    {
        daal::services::daal_free(_rawLineBuffer);
        _rawLineBuffer = NULL;
    }

    /**
     *  Returns a feature manager associated with a File Data Source
     */
    FeatureManager & getFeatureManager() { return _featureManager; }

    size_t getNumericTableNumberOfColumns() DAAL_C11_OVERRIDE { return _featureManager.getNumericTableNumberOfColumns(); }

    services::Status setDictionary(DataSourceDictionary * dict) DAAL_C11_OVERRIDE
    {
        services::Status s = DataSource::setDictionary(dict);
        _featureManager.setFeatureDetailsFromDictionary(dict);

        return s;
    }

    size_t loadDataBlock(NumericTable * nt) DAAL_C11_OVERRIDE
    {
        services::Status s = super::checkDictionary();
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        s = checkInputNumericTable(nt);
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        size_t maxRows     = (_initialMaxRows > 0 ? _initialMaxRows : 10);
        size_t nrows       = 0;
        const size_t ncols = getNumericTableNumberOfColumns();
        DataCollection tables;
        for (;; maxRows *= 2)
        {
            NumericTablePtr ntCurrent = HomogenNumericTable<DAAL_DATA_TYPE>::create(ncols, maxRows, NumericTableIface::doAllocate, &s);
            if (!s)
            {
                this->_status.add(services::throwIfPossible(services::Status(services::ErrorNumericTableNotAllocated)));
                break;
            }
            tables.push_back(ntCurrent);
            const size_t rows = loadDataBlock(maxRows, ntCurrent.get());
            nrows += rows;
            if (rows < maxRows) break;
        }

        s = resetNumericTable(nt, nrows);
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        BlockDescriptor<DAAL_DATA_TYPE> blockCurrent, block;
        nt->getBlockOfRows(0, nrows, writeOnly, block);
        size_t pos = 0;
        int result = 0;
        for (size_t i = 0; i < tables.size(); i++)
        {
            NumericTable * ntCurrent = (NumericTable *)(tables[i].get());
            size_t rows              = ntCurrent->getNumberOfRows();

            if (!rows) continue;

            ntCurrent->getBlockOfRows(0, rows, readOnly, blockCurrent);

            result |= services::internal::daal_memcpy_s(&(block.getBlockPtr()[pos * ncols]), rows * ncols * sizeof(DAAL_DATA_TYPE),
                                                        blockCurrent.getBlockPtr(), rows * ncols * sizeof(DAAL_DATA_TYPE));

            ntCurrent->releaseBlockOfRows(blockCurrent);

            super::combineStatistics(ntCurrent, nt, pos == 0);
            pos += rows;
        }

        nt->releaseBlockOfRows(block);

        if (result)
        {
            this->_status.add(services::throwIfPossible(services::Status(services::ErrorMemoryCopyFailedInternal)));
        }
        return nrows;
    }

    size_t loadDataBlock(size_t maxRows, NumericTable * nt) DAAL_C11_OVERRIDE
    {
        size_t nLines = loadDataBlock(maxRows, 0, maxRows, nt);
        nt->resize(nLines);
        return nLines;
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows, NumericTable * nt) DAAL_C11_OVERRIDE
    {
        services::Status s = super::checkDictionary();
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        s = checkInputNumericTable(nt);
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        if (rowOffset + maxRows > fullRows)
        {
            this->_status.add(services::throwIfPossible(services::ErrorIncorrectDataRange));
            return 0;
        }

        s = resetNumericTable(nt, fullRows);
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }

        if (_parseHeader && !_firstRowRead)
        {
            // Skip header
            s = readLine();
            if (!s)
            {
                this->_status.add(services::throwIfPossible(s));
                return 0;
            }

            _firstRowRead = true;
        }

        size_t j = 0;

        BlockDescriptor<DAAL_DATA_TYPE> ntBlock;
        nt->getBlockOfRows(0, nt->getNumberOfRows(), readWrite, ntBlock);

        for (; j < maxRows && !iseof(); j++)
        {
            s = readLine();
            if (!s)
            {
                this->_status.add(services::throwIfPossible(s));

                return 0;
            }
            if (!_rawLineLength)
            {
                break;
            }

            services::BufferView<DAAL_DATA_TYPE> rowBuffer(ntBlock.getBlockPtr() + (rowOffset + j) * nt->getNumberOfColumns(),
                                                           ntBlock.getNumberOfColumns());

            _featureManager.parseRowIn(_rawLineBuffer, _rawLineLength, this->_dict.get(), rowBuffer, rowOffset + j);

            super::updateStatistics(j, nt, ntBlock.getBlockPtr(), rowOffset);
        }

        nt->releaseBlockOfRows(ntBlock);

        _featureManager.finalize(this->_dict.get());

        return rowOffset + j;
    }

    size_t loadDataBlock() DAAL_C11_OVERRIDE { return DataSource::loadDataBlock(); }

    size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE { return DataSource::loadDataBlock(maxRows); }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows) DAAL_C11_OVERRIDE
    {
        return DataSource::loadDataBlock(maxRows, rowOffset, fullRows);
    }

    services::Status createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        services::Status s;

        if (_dict)
        {
            return services::throwIfPossible(services::Status(services::ErrorDictionaryAlreadyAvailable));
        }

        _dict = DataSourceDictionary::create(&s);
        if (!s)
        {
            return s;
        }
        _contextDictFlag = true;

        if (_parseHeader)
        {
            s = readLine();
            if (!s)
            {
                return services::throwIfPossible(s);
            }
            _featureManager.parseRowAsHeader(_rawLineBuffer, _rawLineLength);
        }

        s = readLine();
        if (!s)
        {
            return services::throwIfPossible(s);
        }
        _featureManager.parseRowAsDictionary(_rawLineBuffer, _rawLineLength, this->_dict.get());

        return services::Status();
    }

    size_t getNumberOfAvailableRows() DAAL_C11_OVERRIDE { return 0; }

protected:
    virtual bool iseof() const          = 0;
    virtual services::Status readLine() = 0;

    virtual services::Status resetNumericTable(NumericTable * nt, const size_t newSize)
    {
        services::Status s;

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        const size_t nFeatures           = getNumericTableNumberOfColumns();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++) ntDict->setFeature((*_dict)[i].ntFeature, i);

        s = super::resizeNumericTableImpl(newSize, nt);
        if (!s)
        {
            return s;
        }

        nt->setNormalizationFlag(NumericTable::nonNormalized);
        return services::Status();
    }

    virtual services::Status checkInputNumericTable(const NumericTable * const nt) const
    {
        if (!nt)
        {
            return services::Status(services::ErrorNullInputNumericTable);
        }

        const NumericTable::StorageLayout layout = nt->getDataLayout();
        if (layout == NumericTable::csrArray)
        {
            return services::Status(services::ErrorIncorrectTypeOfInputNumericTable);
        }

        return services::Status();
    }

    bool enlargeBuffer()
    {
        int newRawLineBufferLen = _rawLineBufferLen * 2;
        char * newRawLineBuffer = (char *)daal::services::daal_malloc(newRawLineBufferLen);
        if (newRawLineBuffer == 0) return false;
        int result = daal::services::internal::daal_memcpy_s(newRawLineBuffer, newRawLineBufferLen, _rawLineBuffer, _rawLineBufferLen);
        if (result)
        {
            this->_status.add(services::throwIfPossible(services::Status(services::ErrorMemoryCopyFailedInternal)));
        }
        daal::services::daal_free(_rawLineBuffer);
        _rawLineBuffer    = newRawLineBuffer;
        _rawLineBufferLen = newRawLineBufferLen;
        return true;
    }

private:
    services::Status initialize(size_t initialMaxRows)
    {
        _parseHeader     = false;
        _firstRowRead    = false;
        _contextDictFlag = false;
        _rawLineLength   = 0;
        _initialMaxRows  = initialMaxRows;

        _rawLineBufferLen = (int)INITIAL_LINE_BUFFER_LENGTH;
        _rawLineBuffer    = (char *)daal::services::daal_malloc(_rawLineBufferLen);
        if (!_rawLineBuffer)
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        return services::Status();
    }

protected:
    char * _rawLineBuffer;
    int _rawLineBufferLen;
    int _rawLineLength;

private:
    bool _parseHeader;
    bool _firstRowRead;
    bool _contextDictFlag;
    FeatureManager _featureManager;

    static const size_t INITIAL_LINE_BUFFER_LENGTH = 1024;
};

/** @} */
} // namespace interface1

using interface1::CsvDataSource;
using interface1::CsvDataSourceOptions;

inline CsvDataSourceOptions::Value operator|(const CsvDataSourceOptions::Value & lhs, const CsvDataSourceOptions::Value & rhs)
{
    return CsvDataSourceOptions::unite(lhs, rhs);
}

} // namespace data_management
} // namespace daal

#endif
