/* file: file_data_source.h */
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

#ifndef __FILE_DATA_SOURCE_H__
#define __FILE_DATA_SOURCE_H__

#include <cstdio>

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__FILEDATASOURCE"></a>
 *  \brief Specifies methods to access data stored in files
 *  \tparam FeatureManager         The type of feature manager that specifies how to extract numerical data from CSV
 *  \tparam SummaryStatisticsType  The floating point type to compute summary statics for numeric table
 */
template <typename FeatureManager, typename SummaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE>
class FileDataSource : public CsvDataSource<FeatureManager, SummaryStatisticsType>
{
private:
    typedef CsvDataSource<FeatureManager, SummaryStatisticsType> super;

protected:
    using super::_rawLineBuffer;
    using super::_rawLineBufferLen;
    using super::_rawLineLength;
    using super::_status;

public:
    /**
     *  Main constructor for a Data Source
     *  \param[in]  fileName                        Name of the file that stores data
     *  \param[in]  doAllocateNumericTable          Flag that specifies whether a Numeric Table
     *                                              associated with a File Data Source is allocated inside the Data Source
     *  \param[in]  doCreateDictionaryFromContext   Flag that specifies whether a Data %Dictionary
     *                                              is created from the context of the File Data Source
     *  \param[in]  initialMaxRows                  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    FileDataSource(const std::string & fileName,
                   DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable    = DataSource::notAllocateNumericTable,
                   DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                   size_t initialMaxRows                                                 = 10)
        : super(doAllocateNumericTable, doCreateDictionaryFromContext, initialMaxRows)
    {
        _status |= initialize(fileName);
    }

    /**
     *  Main constructor for a Data Source
     *  \param[in]  fileName        Name of the file that stores data
     *  \param[in]  options         Options of data source
     *  \param[in]  initialMaxRows  Initial value of maximum number of rows in Numeric Table allocated in loadDataBlock() method
     */
    FileDataSource(const std::string & fileName, CsvDataSourceOptions options, size_t initialMaxRows = 10) : super(options, initialMaxRows)
    {
        _status |= initialize(fileName);
    }

    virtual ~FileDataSource()
    {
        if (_file) fclose(_file);
        daal::services::daal_free(_fileBuffer);
    }

public:
    services::Status createDictionaryFromContext() override
    {
        services::Status s = super::createDictionaryFromContext();
        fseek(_file, 0, SEEK_SET);
        _fileBufferPos = _fileBufferLen;
        return s;
    }

    DataSourceIface::DataSourceStatus getStatus() override { return (iseof() ? DataSourceIface::endOfData : DataSourceIface::readyForLoad); }

protected:
    bool iseof() const override { return (_fileBufferPos == _readedFromFileLen && feof(_file)); }

    bool readLine(char * buffer, int count, int & pos)
    {
        bool bRes = true;
        pos       = 0;
        while (pos + 1 < count)
        {
            if (_fileBufferPos < _readedFromFileLen)
            {
                if (_fileBuffer[_fileBufferPos] == '\0')
                {
                    return false;
                }
                buffer[pos] = _fileBuffer[_fileBufferPos];
                ++pos;
                ++_fileBufferPos;
                if (buffer[pos - 1] == '\n') break;
            }
            else
            {
                if (iseof()) break;
                _fileBufferPos     = 0;
                _readedFromFileLen = (int)fread(_fileBuffer, 1, _fileBufferLen, _file);
                if (ferror(_file))
                {
                    bRes = false;
                    break;
                }
            }
        }
        buffer[pos] = '\0';
        return bRes;
    }

    services::Status readLine() override
    {
        _rawLineLength = 0;
        while (!iseof())
        {
            int readLen = 0;
            if (!readLine(_rawLineBuffer + _rawLineLength, _rawLineBufferLen - _rawLineLength, readLen))
            {
                return services::Status(services::ErrorOnFileRead);
            }

            if (readLen <= 0)
            {
                _rawLineLength = 0;
                break;
            }
            _rawLineLength += readLen;
            if (_rawLineBuffer[_rawLineLength - 1] == '\n' || _rawLineBuffer[_rawLineLength - 1] == '\r')
            {
                while (_rawLineLength > 0 && (_rawLineBuffer[_rawLineLength - 1] == '\n' || _rawLineBuffer[_rawLineLength - 1] == '\r'))
                {
                    _rawLineLength--;
                }
                _rawLineBuffer[_rawLineLength] = '\0';
                break;
            }
            if (!super::enlargeBuffer()) return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return services::Status();
    }

private:
    services::Status initialize(const std::string & fileName)
    {
        _file              = NULL;
        _fileName          = fileName;
        _fileBufferLen     = (int)INITIAL_FILE_BUFFER_LENGTH;
        _fileBufferPos     = _fileBufferLen;
        _fileBuffer        = NULL;
        _readedFromFileLen = 0;
        if (fileName.find('\0') != std::string::npos)
        {
            return services::throwIfPossible(services::ErrorNullByteInjection);
        }
#if (defined(_MSC_VER) && (_MSC_VER >= 1400))
        errno_t error;
        error = fopen_s(&_file, fileName.c_str(), "r");
        if (error != 0 || !_file)
        {
            return services::throwIfPossible(services::ErrorOnFileOpen);
        }
#else
        _file = std::fopen((char *)(fileName.c_str()), "r");
        if (!_file)
        {
            return services::throwIfPossible(services::ErrorOnFileOpen);
        }
#endif
        _fileBuffer = (char *)daal::services::daal_malloc(_fileBufferLen);
        if (!_fileBuffer)
        {
            fclose(_file);
            _file = NULL;
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }
        return services::Status();
    }

protected:
    std::string _fileName;

    FILE * _file;

    char * _fileBuffer;
    int _fileBufferLen;
    int _fileBufferPos;
    int _readedFromFileLen;

private:
    static const size_t INITIAL_FILE_BUFFER_LENGTH = 1048576;
};
/** @} */

} // namespace interface1

using interface1::FileDataSource;

} // namespace data_management
} // namespace daal

#endif
