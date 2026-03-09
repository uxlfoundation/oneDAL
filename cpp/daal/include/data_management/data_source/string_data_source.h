/* file: string_data_source.h */
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
//  Implementation of the string data source class.
//--
*/

#ifndef __STRING_DATA_SOURCE_H__
#define __STRING_DATA_SOURCE_H__

#include "services/daal_memory.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data_source/csv_data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__STRINGDATASOURCE"></a>
 *  \brief Specifies methods to access data stored in byte arrays in the C-string format
 *  \tparam _featureManager     FeatureManager used to get numeric data from file strings
 */
template <typename FeatureManager, typename SummaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE>
class StringDataSource : public CsvDataSource<FeatureManager, SummaryStatisticsType>
{
private:
    typedef CsvDataSource<FeatureManager, SummaryStatisticsType> super;
    typedef data_management::HomogenNumericTable<DAAL_DATA_TYPE> DefaultNumericTableType;

protected:
    using super::_rawLineBuffer;
    using super::_rawLineBufferLen;
    using super::_rawLineLength;
    using super::_status;

public:
    /**
     *  Main constructor for a Data Source
     *  \param[in]  data                            Byte array in the C-string format
     *  \param[in]  doAllocateNumericTable          Flag that specifies whether a Numeric Table
     *                                              associated with a File Data Source is allocated inside the Data Source
     *  \param[in]  doCreateDictionaryFromContext   Flag that specifies whether a Data Dictionary
     *                                              is created from the context of the File Data Source
     *  \param[in]  initialMaxRows                  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    StringDataSource(const byte * data, DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable = DataSource::notAllocateNumericTable,
                     DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                     size_t initialMaxRows                                                 = 10)
        : super(doAllocateNumericTable, doCreateDictionaryFromContext, initialMaxRows), _contextDictFlag(false)
    {
        setData(data);
    }

    /**
     *  Sets a new string as a source for data
     *  \param[in]  data  Byte array in the C-string format
     */
    void setData(const byte * data)
    {
        if (!data)
        {
            _status.add(services::throwIfPossible(services::Status(services::ErrorNullPtr)));
            return;
        }
        _stringBufferPos = 0;
        _stringBuffer    = (char *)data;
    }

    /**
     *  Gets data source string data
     *  \return  Byte array in the C-string format
     */
    const byte * getData() { return (const byte *)(_stringBuffer); }

    /**
     *  Resets a data source string
     */
    void resetData() { _stringBufferPos = 0; }

public:
    services::Status createDictionaryFromContext() override
    {
        services::Status s = super::createDictionaryFromContext();
        _stringBufferPos   = 0;
        return s;
    }

    DataSourceIface::DataSourceStatus getStatus() override { return (iseof() ? DataSourceIface::endOfData : DataSourceIface::readyForLoad); }

protected:
    bool iseof() const override { return (_stringBuffer[_stringBufferPos] == '\0'); }

    int readLine(char * buffer, int count)
    {
        int pos = 0;
        for (; pos < count - 1; pos++)
        {
            buffer[pos] = _stringBuffer[_stringBufferPos + pos];

            if (buffer[pos] == '\0' || buffer[pos] == '\n')
            {
                break;
            }
        }
        if (buffer[pos] == '\n')
        {
            pos++;
        }
        _stringBufferPos += pos;
        buffer[pos] = '\0';
        return pos;
    }

    services::Status readLine() override
    {
        _rawLineLength = 0;
        while (!iseof())
        {
            const int readLen = readLine(_rawLineBuffer + _rawLineLength, (int)(_rawLineBufferLen - _rawLineLength));
            if (readLen <= 0)
            {
                _rawLineLength = 0;
                return services::Status();
            }
            _rawLineLength += readLen;
            if (_rawLineBuffer[_rawLineLength - 1] == '\n' || _rawLineBuffer[_rawLineLength - 1] == '\r')
            {
                while (_rawLineLength > 0 && (_rawLineBuffer[_rawLineLength - 1] == '\n' || _rawLineBuffer[_rawLineLength - 1] == '\r'))
                {
                    _rawLineLength--;
                }
                _rawLineBuffer[_rawLineLength] = '\0';
                return services::Status();
            }
            if (!super::enlargeBuffer()) return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return services::Status();
    }

private:
    char * _stringBuffer;
    size_t _stringBufferPos;

    bool _contextDictFlag;
};
/** @} */
} // namespace interface1
using interface1::StringDataSource;

} // namespace data_management
} // namespace daal
#endif
