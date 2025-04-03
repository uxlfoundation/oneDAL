/* file: odbc_data_source.h */
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
//  Implementation of the ODBC data source class
//--
*/
#ifndef __ODBC_DATA_SOURCE_H__
#define __ODBC_DATA_SOURCE_H__

#include <string>

#include <sql.h>
#include <sqltypes.h>
#include <sqlext.h>

#include "services/daal_memory.h"

#include "data_management/data_source/data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data_source/mysql_feature_manager.h"
#include "data_management/data_source/internal/data_source_options.h"

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
 * @ingroup data_sources
 * @{
 */

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__ODBCDATASOURCEOPTIONS"></a>
 *  \brief Options of ODBC data source
 */
class ODBCDataSourceOptions
{
public:
    enum Value
    {
        byDefault                   = 0,
        allocateNumericTable        = 1 << 0,
        createDictionaryFromContext = 1 << 1
    };

    static ODBCDataSourceOptions::Value unite(const ODBCDataSourceOptions::Value & lhs, const ODBCDataSourceOptions::Value & rhs)
    {
        return internal::DataSourceOptionsImpl<Value>::unite(lhs, rhs);
    }

    ODBCDataSourceOptions(Value flags = byDefault) : _impl(flags) {}

    DataSource::NumericTableAllocationFlag getNumericTableAllocationFlag() const
    {
        return (_impl.getFlag(allocateNumericTable)) ? DataSource::doAllocateNumericTable : DataSource::notAllocateNumericTable;
    }

    DataSource::DictionaryCreationFlag getDictionaryCreationFlag() const
    {
        return (_impl.getFlag(createDictionaryFromContext)) ? DataSource::doDictionaryFromContext : DataSource::notDictionaryFromContext;
    }

private:
    internal::DataSourceOptionsImpl<Value> _impl;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__ODBCDATASOURCE"></a>
 * \brief Connects to data sources with the ODBC API
 * \tparam FeatureManager         Type of a data source, supports only \ref SQLFeatureManager
 * \tparam SummaryStatisticsType  The floating point type to compute summary statics for numeric table
 */
template <typename FeatureManager, typename SummaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE>
class ODBCDataSource : public DataSourceTemplate<data_management::HomogenNumericTable<DAAL_DATA_TYPE>, SummaryStatisticsType>
{
private:
    typedef data_management::HomogenNumericTable<DAAL_DATA_TYPE> DefaultNumericTableType;
    typedef DataSourceTemplate<DefaultNumericTableType, SummaryStatisticsType> super;

protected:
    using super::_dict;
    using super::_spnt;
    using super::_initialMaxRows;
    using super::_autoNumericTableFlag;
    using super::_autoDictionaryFlag;
    using super::_status;

public:
    /**
     * Constructor for the ODBCDataSource class
     * \param[in] dbname                Data Source Name  as  configured in settings of the ODBC driver
     * \param[in] tableName             Name of a table to export from a data source
     * \param[in] userName              (optional) Username for the data source
     * \param[in] password              (optional) Password for the Username in the data source
     * \param[in] doAllocateNumericTable        (optional) Flag that specifies whether a Numeric Table
     *                                                     associated with an ODBC Data Source is allocated inside the Data Source
     * \param[in] doCreateDictionaryFromContext (optional) Flag that specifies whether a Data Dictionary
     *                                                     is created from the context of the ODBC Data Source
     * \param[in]  initialMaxRows                          Initial value of maximum number of rows in Numeric Table allocated in
     *                                                     loadDataBlock() method
     */
    ODBCDataSource(const std::string & dbname, const std::string & tableName = "", const std::string & userName = "",
                   const std::string & password                                          = "",
                   DataSourceIface::NumericTableAllocationFlag doAllocateNumericTable    = DataSource::notAllocateNumericTable,
                   DataSourceIface::DictionaryCreationFlag doCreateDictionaryFromContext = DataSource::notDictionaryFromContext,
                   size_t initialMaxRows                                                 = 10)
        : super(doAllocateNumericTable, doCreateDictionaryFromContext)
    {
        initialize(initialMaxRows);
        if (dbname.find('\0') != std::string::npos || tableName.find('\0') != std::string::npos || userName.find('\0') != std::string::npos
            || password.find('\0') != std::string::npos)
        {
            this->_status.add(services::throwIfPossible(services::ErrorNullByteInjection));
            return;
        }
        _status |= connectUsingUserNameAndPassword(dbname, userName, password);
        if (!_status)
        {
            return;
        }
        _status |= executeSelectAllQuery(tableName);
    }

    /**
     * Constructor for the ODBCDataSource class
     * \param[in] dbname           Data Source Name  as  configured in settings of the ODBC driver
     * \param[in] tableName        Name of a table to export from a data source
     * \param[in] userName         Username for the data source
     * \param[in] password         Password for the Username in the data source
     * \param[in]  options         The options of ODBC Data Source
     * \param[in]  initialMaxRows  Initial value of maximum number of rows in Numeric Table allocated in
     *                             loadDataBlock() method
     */
    ODBCDataSource(const std::string & dbname, const std::string & tableName, const std::string & userName, const std::string & password,
                   const ODBCDataSourceOptions & options, size_t initialMaxRows = 10)
        : super(options.getNumericTableAllocationFlag(), options.getDictionaryCreationFlag())
    {
        initialize(initialMaxRows);
        if (dbname.find('\0') != std::string::npos || tableName.find('\0') != std::string::npos || userName.find('\0') != std::string::npos
            || password.find('\0') != std::string::npos)
        {
            this->_status.add(services::throwIfPossible(services::ErrorNullByteInjection));
            return;
        }
        _status |= connectUsingUserNameAndPassword(dbname, userName, password);
        if (!_status)
        {
            return;
        }
        _status |= executeSelectAllQuery(tableName);
    }

    /**
     * Constructor for the ODBCDataSource class
     * \param[in]  connectionString  The connection string to ODBC Driver
     * \param[in]  options           The options of ODBC Data Source
     * \param[in]  initialMaxRows    Initial value of maximum number of rows in Numeric Table allocated in
     *                               loadDataBlock() method
     */
    ODBCDataSource(const std::string & connectionString, const ODBCDataSourceOptions & options, size_t initialMaxRows = 10)
        : super(options.getNumericTableAllocationFlag(), options.getDictionaryCreationFlag())
    {
        initialize(initialMaxRows);
        if (connectionString.find('\0') != std::string::npos)
        {
            this->_status.add(services::throwIfPossible(services::ErrorNullByteInjection));
            return;
        }
        _status |= connectUsingConnectionString(connectionString);
    }

    virtual ~ODBCDataSource() { freeHandlesInternal(); }

    services::Status executeQuery(const std::string & query)
    {
        if (query.find('\0') != std::string::npos)
        {
            this->_status.add(services::throwIfPossible(services::ErrorNullByteInjection));
            return _status;
        }
        _idxLastRead = 0;

        if (_autoNumericTableFlag == DataSource::doAllocateNumericTable)
        {
            _spnt.reset();
        }

        if (_autoDictionaryFlag == DataSource::doDictionaryFromContext)
        {
            _dict.reset();
        }

        if (_hdlStmt)
        {
            SQLRETURN ret = SQLFreeHandle(SQL_HANDLE_STMT, _hdlStmt);
            if (!SQL_SUCCEEDED(ret))
            {
                return services::throwIfPossible(services::ErrorSQLstmtHandle);
            }
            _hdlStmt = SQL_NULL_HSTMT;
        }

        SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_STMT, _hdlDbc, &_hdlStmt);
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorSQLstmtHandle);
        }

        ret = SQLExecDirect(_hdlStmt, (SQLCHAR *)query.c_str(), SQL_NTS);
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorODBC);
        }

        _connectionStatus = DataSource::readyForLoad;
        return services::Status();
    }

    /**
     *  Frees ODBC connection handles
     */
    services::Status freeHandles()
    {
        SQLRETURN ret = freeHandlesInternal();
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorODBC);
        }

        _connectionStatus = DataSource::notReady;
        return services::Status();
    }

    virtual size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE
    {
        services::Status s = checkConnection();
        if (!s)
        {
            return 0;
        }

        s = super::checkDictionary();
        if (!s)
        {
            return 0;
        }

        s = super::checkNumericTable();
        if (!s)
        {
            return 0;
        }

        return loadDataBlock(maxRows, _spnt.get());
    }

    /**
     *  Loads a data block of a specified size into an externally allocated Numeric Table
     *  \param[in] maxRows Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param nt Externally allocated Numeric Table
     *  \return Actual number of rows loaded from the Data Source
     */
    virtual size_t loadDataBlock(size_t maxRows, NumericTable * nt)
    {
        services::Status s = checkConnection();
        if (!s)
        {
            return 0;
        }

        s = super::checkDictionary();
        if (!s)
        {
            return 0;
        }

        if (nt == NULL)
        {
            this->_status.add(services::throwIfPossible(services::ErrorNullInputNumericTable));
            return 0;
        }

        super::resizeNumericTableImpl(maxRows, nt);

        if (nt->getDataMemoryStatus() == NumericTableIface::userAllocated)
        {
            if (nt->getNumberOfRows() < maxRows)
            {
                this->_status.add(services::throwIfPossible(services::ErrorIncorrectNumberOfObservations));
                return 0;
            }
            if (nt->getNumberOfColumns() != _dict->getNumberOfFeatures())
            {
                this->_status.add(services::throwIfPossible(services::ErrorIncorrectNumberOfFeatures));
                return 0;
            }
        }

        _connectionStatus = _featureManager.statementResultsNumericTable(_hdlStmt, nt, maxRows);

        size_t nRead = nt->getNumberOfRows();
        _idxLastRead += nRead;

        BlockDescriptor<DAAL_DATA_TYPE> blockNt;
        nt->getBlockOfRows(0, nt->getNumberOfRows(), readOnly, blockNt);
        DAAL_DATA_TYPE * row = blockNt.getBlockPtr();

        if (nt->basicStatistics.get(NumericTableIface::minimum).get() != NULL && nt->basicStatistics.get(NumericTableIface::maximum).get() != NULL
            && nt->basicStatistics.get(NumericTableIface::sum).get() != NULL && nt->basicStatistics.get(NumericTableIface::sumSquares).get() != NULL)
        {
            for (size_t i = 0; i < nRead; i++)
            {
                super::updateStatistics(i, nt, row);
            }
        }

        nt->releaseBlockOfRows(blockNt);

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        size_t nFeatures                 = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return nRead;
    }

    size_t loadDataBlock() DAAL_C11_OVERRIDE
    {
        services::Status s;

        s = checkConnection();
        if (!s)
        {
            return 0;
        }

        s = super::checkDictionary();
        if (!s)
        {
            return 0;
        }

        s = super::checkNumericTable();
        if (!s)
        {
            return 0;
        }

        return loadDataBlock(_spnt.get());
    }

    size_t loadDataBlock(NumericTable * nt) DAAL_C11_OVERRIDE
    {
        services::Status s;

        s = checkConnection();
        if (!s)
        {
            return 0;
        }

        s = super::checkDictionary();
        if (!s)
        {
            return 0;
        }

        if (nt == NULL)
        {
            this->_status.add(services::throwIfPossible(services::ErrorNullInputNumericTable));
            return 0;
        }

        size_t maxRows = (_initialMaxRows > 0 ? _initialMaxRows : 10);
        size_t nrows   = 0;
        size_t ncols   = _dict->getNumberOfFeatures();

        DataCollection tables;

        for (;;)
        {
            NumericTablePtr ntCurrent = HomogenNumericTable<DAAL_DATA_TYPE>::create(ncols, maxRows, NumericTableIface::doAllocate, &s);
            if (!s)
            {
                this->_status.add(services::throwIfPossible(services::ErrorNumericTableNotAllocated));
                break;
            }
            tables.push_back(ntCurrent);
            size_t rows = loadDataBlock(maxRows, ntCurrent.get());
            nrows += rows;
            if (rows < maxRows)
            {
                break;
            }
            maxRows *= 2;
        }

        super::resizeNumericTableImpl(nrows, nt);
        nt->setNormalizationFlag(NumericTable::nonNormalized);

        BlockDescriptor<DAAL_DATA_TYPE> blockCurrent, block;

        size_t pos = 0;
        int result = 0;

        for (size_t i = 0; i < tables.size(); i++)
        {
            NumericTable * ntCurrent = (NumericTable *)(tables[i].get());
            size_t rows              = ntCurrent->getNumberOfRows();

            if (rows == 0)
            {
                continue;
            }

            ntCurrent->getBlockOfRows(0, rows, readOnly, blockCurrent);
            nt->getBlockOfRows(pos, rows, writeOnly, block);

            result |= services::internal::daal_memcpy_s(block.getBlockPtr(), rows * ncols * sizeof(DAAL_DATA_TYPE), blockCurrent.getBlockPtr(),
                                                        rows * ncols * sizeof(DAAL_DATA_TYPE));

            ntCurrent->releaseBlockOfRows(blockCurrent);
            nt->releaseBlockOfRows(block);

            super::combineStatistics(ntCurrent, nt, pos == 0);
            pos += rows;
        }
        if (result)
        {
            this->_status.add(services::throwIfPossible(services::ErrorMemoryCopyFailedInternal));
        }

        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        size_t nFeatures                 = _dict->getNumberOfFeatures();
        ntDict->setNumberOfFeatures(nFeatures);
        for (size_t i = 0; i < nFeatures; i++)
        {
            ntDict->setFeature((*_dict)[i].ntFeature, i);
        }

        return nrows;
    }

    services::Status createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        services::Status status = checkConnection();
        DAAL_CHECK_STATUS_VAR(status);

        _connectionStatus = DataSource::notReady;

        if (_dict)
        {
            return services::throwIfPossible(services::ErrorDictionaryAlreadyAvailable);
        }

        _dict = DataSourceDictionary::create(&status);
        DAAL_CHECK_STATUS_VAR(status);

        status |= _featureManager.createDictionary(_hdlStmt, _dict.get());
        DAAL_CHECK_STATUS_VAR(status);

        _connectionStatus = DataSource::readyForLoad;
        return status;
    }

    DataSourceIface::DataSourceStatus getStatus() DAAL_C11_OVERRIDE { return _connectionStatus; }

    size_t getNumberOfAvailableRows() DAAL_C11_OVERRIDE { return 0; }

    FeatureManager & getFeatureManager() { return _featureManager; }

private:
    void initialize(size_t initialMaxRows)
    {
        _hdlDbc  = SQL_NULL_HDBC;
        _hdlEnv  = SQL_NULL_HENV;
        _hdlStmt = SQL_NULL_HSTMT;

        _idxLastRead      = 0;
        _initialMaxRows   = initialMaxRows;
        _connectionStatus = DataSource::notReady;
    }

    services::Status connectUsingUserNameAndPassword(const std::string & dbname, const std::string & username, const std::string & password)
    {
        SQLRETURN ret = setupHandlesInternal();
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorHandlesSQL);
        }

        ret = connectInternal(dbname, username, password);
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorODBC);
        }

        return services::Status();
    }

    services::Status connectUsingConnectionString(const std::string & connectionString)
    {
        SQLRETURN ret = setupHandlesInternal();
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorHandlesSQL);
        }

        ret = connectDriverInternal(connectionString);
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorODBC);
        }

        return services::Status();
    }

    services::Status executeSelectAllQuery(const std::string & tableName)
    {
        if (!tableName.empty())
        {
            return executeQuery("SELECT * FROM " + tableName);
        }
        return services::Status();
    }

    SQLRETURN connectInternal(const std::string & dbname, const std::string & username, const std::string & password)
    {
        return SQLConnect(_hdlDbc, (SQLCHAR *)dbname.c_str(), (SQLSMALLINT)dbname.size(), (SQLCHAR *)username.c_str(), (SQLSMALLINT)username.size(),
                          (SQLCHAR *)password.c_str(), (SQLSMALLINT)password.size());
    }

    SQLRETURN connectDriverInternal(const std::string & connectionString)
    {
        SQLSMALLINT outConnectionStringLength;
        return SQLDriverConnect(_hdlDbc, SQL_NULL_HANDLE, (SQLCHAR *)connectionString.c_str(), (SQLSMALLINT)connectionString.size(), (SQLCHAR *)NULL,
                                (SQLSMALLINT)0, &outConnectionStringLength, SQL_DRIVER_NOPROMPT);
    }

    SQLRETURN setupHandlesInternal()
    {
        SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &_hdlEnv);
        if (!SQL_SUCCEEDED(ret))
        {
            return ret;
        }

        ret = SQLSetEnvAttr(_hdlEnv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, SQL_IS_UINTEGER);
        if (!SQL_SUCCEEDED(ret))
        {
            return ret;
        }

        ret = SQLAllocHandle(SQL_HANDLE_DBC, _hdlEnv, &_hdlDbc);
        if (!SQL_SUCCEEDED(ret))
        {
            return ret;
        }

        return SQL_SUCCESS;
    }

    SQLRETURN freeHandlesInternal()
    {
        if (_hdlDbc == SQL_NULL_HDBC || _hdlEnv == SQL_NULL_HENV)
        {
            return SQL_SUCCESS;
        }

        SQLRETURN ret = SQLDisconnect(_hdlDbc);
        if (!SQL_SUCCEEDED(ret))
        {
            return ret;
        }

        ret = SQLFreeHandle(SQL_HANDLE_DBC, _hdlDbc);
        if (!SQL_SUCCEEDED(ret))
        {
            return ret;
        }

        ret = SQLFreeHandle(SQL_HANDLE_ENV, _hdlEnv);
        if (!SQL_SUCCEEDED(ret))
        {
            return ret;
        }

        _hdlDbc = SQL_NULL_HDBC;
        _hdlEnv = SQL_NULL_HENV;

        return SQL_SUCCESS;
    }

    services::Status checkConnection()
    {
        if (_connectionStatus == DataSource::notReady)
        {
            return services::throwIfPossible(services::ErrorSourceDataNotAvailable);
        }

        return services::Status();
    }

private:
    size_t _idxLastRead;
    FeatureManager _featureManager;
    DataSourceIface::DataSourceStatus _connectionStatus;

    SQLHENV _hdlEnv;
    SQLHDBC _hdlDbc;
    SQLHSTMT _hdlStmt;
};
/** @} */

} // namespace interface1

using interface1::ODBCDataSource;
using interface1::ODBCDataSourceOptions;

inline ODBCDataSourceOptions::Value operator|(const ODBCDataSourceOptions::Value & lhs, const ODBCDataSourceOptions::Value & rhs)
{
    return ODBCDataSourceOptions::unite(lhs, rhs);
}

} // namespace data_management
} // namespace daal

#endif
