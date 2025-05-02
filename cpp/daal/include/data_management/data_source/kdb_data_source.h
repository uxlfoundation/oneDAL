/* file: kdb_data_source.h */
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
//  Implementation of the KDB data source class
//--
*/
#ifndef __KDB_DATA_SOURCE_H__
#define __KDB_DATA_SOURCE_H__

#include <sstream>
#include <fstream>
#include "services/daal_memory.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

#include <k.h>

#include "data_management/data_source/kdb_feature_manager.h"

namespace daal
{
namespace data_management
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-KDBDATASOURCE"></a>
 * \brief Connects to data sources with the KDB API.
 *
 * \tparam _featureManager       Type of a data source, supports only \ref KDBFeatureManager
 */

template <typename _featureManager, typename summaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE>
class KDBDataSource : public DataSourceTemplate<data_management::HomogenNumericTable<DAAL_DATA_TYPE>, summaryStatisticsType>
{
public:
    typedef _featureManager FeatureManager;

    using DataSourceIface::NumericTableAllocationFlag;
    using DataSourceIface::DictionaryCreationFlag;
    using DataSourceIface::DataSourceStatus;

    using DataSource::checkDictionary;
    using DataSource::checkNumericTable;
    using DataSource::freeNumericTable;
    using DataSource::_dict;
    using DataSource::_initialMaxRows;

protected:
    typedef data_management::HomogenNumericTable<DAAL_DATA_TYPE> DefaultNumericTableType;

    FeatureManager featureManager;

public:
    /**
     * Constructor for the KDBDataSource class
     *
     * \param[in] dbname                Data Source Name  as  configured in settings of the KDB driver
     * \param[in] port                  Connection port number
     * \param[in] tablename             Name of a table to export from a data source
     * \param[in] username              (optional) Username for the data source
     * \param[in] password              (optional) Password for the Username in the data source
     * \param[in] doAllocateNumericTable        (optional) Flag that specifies whether a Numeric Table
     *                                                     associated with an KDB Data Source is allocated inside the Data Source
     * \param[in] doCreateDictionaryFromContext (optional) Flag that specifies whether a Data Dictionary
     *                                                     is created from the context of the KDB Data Source
     * \param[in]  initialMaxRows                          Initial value of maximum number of rows in Numeric Table allocated in
     *                                                     loadDataBlock() method
     *
     */
    KDBDataSource(const std::string & dbname, size_t port, const std::string & tablename, const std::string & username = "",
                  const std::string & password                                          = "",
                  DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable    = DataSource::notAllocateNumericTable,
                  DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                  size_t initialMaxRows                                                 = 10)
        : DataSourceTemplate<DefaultNumericTableType, summaryStatisticsType>(doAllocateNumericTable, doCreateDictionaryFromContext),
          _port(port),
          _idx_last_read(0)
    {
        if (dbname.find('\0') != std::string::npos || tablename.find('\0') != std::string::npos || username.find('\0') != std::string::npos
            || password.find('\0') != std::string::npos)
        {
            this->_errors->add(services::ErrorNullByteInjection);
            return;
        }
        _dbname         = dbname;
        _username       = username;
        _password       = password;
        _tablename      = tablename;
        _query          = _tablename;
        _initialMaxRows = initialMaxRows;
    }

    /*! \private */
    ~KDBDataSource() {}

    size_t loadDataBlock() DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if (this->_errors->size() != 0)
        {
            return 0;
        }

        checkNumericTable();
        if (this->_errors->size() != 0)
        {
            return 0;
        }

        return loadDataBlock(0, this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(NumericTable * nt) DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if (this->_errors->size() != 0)
        {
            return 0;
        }

        return loadDataBlock(0, nt);
    }

    virtual size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE
    {
        checkDictionary();
        if (!this->_errors->isEmpty())
        {
            return 0;
        }

        checkNumericTable();
        if (!this->_errors->isEmpty())
        {
            return 0;
        }

        return loadDataBlock(maxRows, this->DataSource::_spnt.get());
    }

    /**
     *  Loads a data block of a specified size into an externally allocated Numeric Table
     *  \param[in] maxRows Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param nt Externally allocated Numeric Table
     *  \return Actual number of rows loaded from the Data Source
     */
    virtual size_t loadDataBlock(size_t maxRows, NumericTable * nt)
    {
        checkDictionary();

        if (this->_errors->size() != 0)
        {
            return 0;
        }

        if (nt == NULL)
        {
            this->_errors->add(services::ErrorNullInputNumericTable);
            return 0;
        }

        I handle = _kdbConnect();

        if (handle <= 0)
        {
            return 0;
        }

        size_t nRows = getNumberOfAvailableRows();

        if (nRows == 0)
        {
            DataSourceTemplate<DefaultNumericTableType, summaryStatisticsType>::resizeNumericTableImpl(0, nt);
            _kdbClose(handle);
            return 0;
        }

        if (maxRows != 0 && nRows > maxRows)
        {
            nRows = maxRows;
        }

        std::ostringstream query;
        query << "(" << _query << ")[(til " << nRows << ") + " << _idx_last_read << +"]";
        std::string query_exec = query.str();

        K result = k(handle, const_cast<char *>(query_exec.c_str()), (K)0);

        _kdbClose(handle);

        _idx_last_read += nRows;

        DataSourceTemplate<DefaultNumericTableType, summaryStatisticsType>::resizeNumericTableImpl(nRows, nt);

        if (nt->getDataMemoryStatus() == NumericTableIface::userAllocated)
        {
            if (nt->getNumberOfRows() < nRows)
            {
                r0(result);
                this->_errors->add(services::ErrorIncorrectNumberOfObservations);
                return 0;
            }
            if (nt->getNumberOfColumns() != _dict->getNumberOfFeatures())
            {
                r0(result);
                this->_errors->add(services::ErrorIncorrectNumberOfFeatures);
                return 0;
            }
        }

        if (result->t == XT)
        {
            K columnData = kK(result->k)[1];
            featureManager.statementResultsNumericTableFromColumnData(columnData, nt, nRows);
        }
        else if (result->t == XD)
        {
            K columnData = kK(result)[1];
            featureManager.statementResultsNumericTableFromColumnData(columnData, nt, nRows);
        }
        else
        {
            featureManager.statementResultsNumericTableFromList(result, nt, nRows);
        }
        r0(result);

        if (nt->basicStatistics.get(NumericTableIface::minimum).get() != NULL && nt->basicStatistics.get(NumericTableIface::maximum).get() != NULL
            && nt->basicStatistics.get(NumericTableIface::sum).get() != NULL && nt->basicStatistics.get(NumericTableIface::sumSquares).get() != NULL)
        {
            BlockDescriptor<DAAL_DATA_TYPE> blockNt;
            nt->getBlockOfRows(0, nt->getNumberOfRows(), readOnly, blockNt);
            DAAL_DATA_TYPE * row = blockNt.getBlockPtr();

            for (size_t i = 0; i < nRows; i++)
            {
                DataSourceTemplate<DefaultNumericTableType, summaryStatisticsType>::updateStatistics(i, nt, row);
            }

            nt->releaseBlockOfRows(blockNt);
        }

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        size_t nFeatures                 = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return nRows;
    }

    services::Status createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        if (_dict) return services::Status(services::ErrorDictionaryAlreadyAvailable);

        I handle = _kdbConnect();

        std::string query_exec = "(" + _query + ")[til 1]";

        K result = k(handle, const_cast<char *>(query_exec.c_str()), (K)0);

        if (!result)
        {
            _kdbClose(handle);
            return services::Status(services::ErrorKDBNetworkError);
        }

        if (result->t == -128)
        {
            r0(result);
            _kdbClose(handle);
            return services::Status(services::ErrorKDBServerError);
        }

        services::Status status;
        _dict = DataSourceDictionary::create(&status);
        if (!status) return status;

        if (result->t == XT)
        {
            featureManager.createDictionaryFromTable(result->k, this->_dict.get());
        }
        else if (result->t == XD)
        {
            featureManager.createDictionaryFromTable(result, this->_dict.get());
        }
        else
        {
            featureManager.createDictionaryFromList(kK(result)[0], this->_dict.get());
        }
        r0(result);

        _kdbClose(handle);
        return status;
    }

    DataSourceIface::DataSourceStatus getStatus() DAAL_C11_OVERRIDE { return DataSourceIface::readyForLoad; }

    size_t getNumberOfAvailableRows() DAAL_C11_OVERRIDE
    {
        I handle = _kdbConnect();

        if (handle <= 0) return 0;

        std::string query_exec = "count " + _query;

        K result = k(handle, const_cast<char *>(query_exec.c_str()), (K)0);

        if (result->t != -KJ)
        {
            this->_errors->add(services::ErrorKDBWrongTypeOfOutput);
            r0(result);
            _kdbClose(handle);
            return 0;
        }

        size_t nRows = result->j;

        r0(result);

        _kdbClose(handle);

        return nRows - _idx_last_read;
    }

    FeatureManager & getFeatureManager() { return featureManager; }

private:
    std::string _dbname;
    size_t _port;
    std::string _username;
    std::string _password;
    std::string _tablename;
    std::string _query;
    size_t _idx_last_read;

    I _kdbConnect()
    {
        I handle = khpu(const_cast<char *>(_dbname.c_str()), _port, const_cast<char *>((_username + ":" + _password).c_str()));

        if (handle < 0)
        {
            this->_errors->add(services::ErrorKDBNoConnection);
            return handle;
        }

        if (handle == 0)
        {
            this->_errors->add(services::ErrorKDBWrongCredentials);
            return handle;
        }

        return handle;
    }

    void _kdbClose(I handle) { kclose(handle); }
};
} // namespace interface1
using interface1::KDBDataSource;

} // namespace data_management
} // namespace daal
#endif
