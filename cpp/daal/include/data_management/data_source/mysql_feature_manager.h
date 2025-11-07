/* file: mysql_feature_manager.h */
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
//  Implementation of the MYSQL data source class.
//--
*/
#ifndef __MYSQL_FEATURE_MANAGER_H__
#define __MYSQL_FEATURE_MANAGER_H__

#include <sstream>

#include "data_management/data/numeric_table.h"
#include "data_management/features/shortcuts.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data_source/internal/sql_feature_utils.h"
#include "data_management/data_source/modifiers/sql/shortcuts.h"
#include "data_management/data_source/modifiers/sql/internal/engine.h"

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
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__SQLFEATUREMANAGER"></a>
 * \brief Interprets the response of SQL data base and fill provided numeric table and dictionary
 * \DAAL_DEPRECATED
 */
class SQLFeatureManager
{
public:
    DAAL_DEPRECATED SQLFeatureManager() : _fetchBuffer(), _errors(services::SharedPtr<services::ErrorCollection>(new services::ErrorCollection)) {}

    /**
     * Adds extended feature modifier
     * \param[in]   featureIds The identifiers of the features to be modified
     * \param[in]   modifier   The feature modifier
     * \param[out]  status     (optional) The pointer to status object
     * \return Reference to itself
     */
    SQLFeatureManager & addModifier(const features::FeatureIdCollectionIfacePtr & featureIds,
                                    const modifiers::sql::FeatureModifierIfacePtr & modifier, services::Status * status = NULL)
    {
        services::Status localStatus;
        if (!_modifiersManager)
        {
            _modifiersManager = modifiers::sql::internal::ModifiersManager::create(&localStatus);
            if (!localStatus)
            {
                services::internal::tryAssignStatusAndThrow(status, localStatus);
                return *this;
            }
        }

        localStatus |= _modifiersManager->addModifier(featureIds, modifier);
        if (!localStatus)
        {
            services::internal::tryAssignStatusAndThrow(status, localStatus);
            return *this;
        }

        return *this;
    }

    /**
     *  Executes an SQL statement from an ODBC statement handle and writes it to a Numeric Table
     *  \param[in]   hdlStmt ODBC statement handle that contains an SQL query
     *  \param[out]  nt      Numeric Table to store query results
     *  \param[in]   maxRows Maximum number of rows that can be read
     */
    DataSourceIface::DataSourceStatus statementResultsNumericTable(SQLHSTMT hdlStmt, NumericTable * nt, size_t maxRows)
    {
        DAAL_ASSERT(nt);
        DAAL_ASSERT(hdlStmt);

        nt->resize(maxRows);

        nt->getBlockOfRows(0, maxRows, writeOnly, _currentRowBlock);
        DAAL_DATA_TYPE * ntBuffer = _currentRowBlock.getBlockPtr();
        const size_t nColumns     = _currentRowBlock.getNumberOfColumns();

        SQLRETURN ret;
        size_t read = 0;
        while (SQL_SUCCEEDED(ret = SQLFetchScroll(hdlStmt, SQL_FETCH_NEXT, 1)))
        {
            services::BufferView<DAAL_DATA_TYPE> rowBuffer(ntBuffer + read * nColumns, nColumns);

            if (_modifiersManager)
            {
                _modifiersManager->applyModifiers(rowBuffer);
            }
            else
            {
                _fetchBuffer->copyTo(rowBuffer);
            }

            read++;
            if (read >= maxRows)
            {
                break;
            }
        }

        nt->releaseBlockOfRows(_currentRowBlock);
        nt->resize(read);

        DataSourceIface::DataSourceStatus status = DataSourceIface::readyForLoad;
        if (ret != SQL_NO_DATA)
        {
            if (!SQL_SUCCEEDED(ret))
            {
                status = DataSourceIface::notReady;
                _errors->add(services::ErrorODBC);
            }
        }
        else
        {
            if (read < maxRows)
            {
                status = DataSourceIface::endOfData;
            }
        }
        return status;
    }

    /**
     *  Creates a data dictionary from an ODBC statement handle
     *  \param[in]   hdlStmt     ODBC statement handle that contains an SQL query
     *  \param[out]  dictionary  Dictionary to be created
     */
    services::Status createDictionary(SQLHSTMT hdlStmt, DataSourceDictionary * dictionary)
    {
        DAAL_ASSERT(dictionary);
        DAAL_ASSERT(hdlStmt);

        services::Status status;

        const internal::SQLFeaturesInfo & featuresInfo = getFeaturesInfo(hdlStmt, &status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(status, bindSQLColumns(hdlStmt, featuresInfo));
        DAAL_CHECK_STATUS(status, fillDictionary(*dictionary, featuresInfo));

        return status;
    }

    /**
     *  Limits the SELECT query.
     *  \param[in]   query         Query to be limited
     *  \param[in]   idx_last_read Index of the first row to read
     *  \param[in]   maxRows       Maximum number of rows to read
     *  \return Full limited query with the ';' symbol at the end
     */
    std::string setLimitQuery(std::string & query, size_t idx_last_read, size_t maxRows)
    {
        if (query.find('\0') != std::string::npos)
        {
            this->_errors->add(services::ErrorNullByteInjection);
            return std::string();
        }
        std::stringstream ss;
        ss << query << " LIMIT " << idx_last_read << ", " << maxRows << ";";
        return ss.str();
    }

    services::ErrorCollectionPtr getErrors() { return services::ErrorCollectionPtr(new services::ErrorCollection()); }

private:
    internal::SQLFeaturesInfo getFeaturesInfo(SQLHSTMT hdlStmt, services::Status * status = NULL)
    {
        SQLSMALLINT nFeatures = 0;
        SQLRETURN ret         = SQLNumResultCols(hdlStmt, &nFeatures);
        if (!SQL_SUCCEEDED(ret))
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorODBC);
            return internal::SQLFeaturesInfo();
        }

        internal::SQLFeaturesInfo featuresInfo;

        for (int i = 0; i < nFeatures; i++)
        {
            const int bufferSize = 128;
            char label[bufferSize];

            SQLLEN sqlType;
            SQLLEN sqlOctetLength;
            SQLSMALLINT labelLenUsed;

            SQLLEN sqlIsUnsigned;
            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)(i + 1), SQL_DESC_UNSIGNED, NULL, 0, NULL, &sqlIsUnsigned);
            if (!SQL_SUCCEEDED(ret))
            {
                services::internal::tryAssignStatusAndThrow(status, services::ErrorODBC);
                return internal::SQLFeaturesInfo();
            }

            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)(i + 1), SQL_DESC_TYPE, NULL, 0, NULL, &sqlType);
            if (!SQL_SUCCEEDED(ret))
            {
                services::internal::tryAssignStatusAndThrow(status, services::ErrorODBC);
                return internal::SQLFeaturesInfo();
            }

            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)(i + 1), SQL_DESC_OCTET_LENGTH, NULL, 0, NULL, &sqlOctetLength);
            if (!SQL_SUCCEEDED(ret))
            {
                services::internal::tryAssignStatusAndThrow(status, services::ErrorODBC);
                return internal::SQLFeaturesInfo();
            }

            ret = SQLColAttributes(hdlStmt, (SQLUSMALLINT)(i + 1), SQL_DESC_NAME, (SQLPOINTER)label, (SQLSMALLINT)bufferSize, &labelLenUsed, NULL);
            if (!SQL_SUCCEEDED(ret))
            {
                services::internal::tryAssignStatusAndThrow(status, services::ErrorODBC);
                return internal::SQLFeaturesInfo();
            }
            services::Status internalStatus = services::internal::checkForNullByteInjection(label, label + labelLenUsed);
            if (!internalStatus)
            {
                services::internal::tryAssignStatusAndThrow(status, internalStatus);
                return internal::SQLFeaturesInfo();
            }
            const services::String labelStr(label);
            const bool isSigned = sqlIsUnsigned == SQL_FALSE;

            featuresInfo.add(internal::SQLFeatureInfo(labelStr, sqlType, sqlOctetLength, isSigned));
        }

        return featuresInfo;
    }

    services::Status bindSQLColumns(SQLHSTMT hdlStmt, const internal::SQLFeaturesInfo & featuresInfo)
    {
        DAAL_ASSERT(hdlStmt);

        services::Status status;

        const internal::SQLFetchMode::Value fetchMode =
            _modifiersManager ? internal::SQLFetchMode::useNativeSQLTypes : internal::SQLFetchMode::castToFloatingPointType;
        _fetchBuffer = internal::SQLFetchBuffer::create(featuresInfo, fetchMode, &status);
        DAAL_CHECK_STATUS_VAR(status);

        SQLRETURN ret = SQLFreeStmt(hdlStmt, SQL_UNBIND);
        if (!SQL_SUCCEEDED(ret))
        {
            return services::throwIfPossible(services::ErrorODBC);
        }

        const SQLSMALLINT targetSQLType = internal::SQLFetchMode::getTargetType(fetchMode);
        for (size_t i = 0; i < featuresInfo.getNumberOfFeatures(); i++)
        {
            char * const buffer             = _fetchBuffer->getBufferForFeature(i);
            const SQLLEN bufferSize         = _fetchBuffer->getBufferSizeForFeature(i);
            SQLLEN * const actualSizeBuffer = _fetchBuffer->getActualDataSizeBufferForFeature(i);

            ret = SQLBindCol(hdlStmt, (SQLUSMALLINT)(i + 1), targetSQLType, (SQLPOINTER)buffer, bufferSize, actualSizeBuffer);
            if (!SQL_SUCCEEDED(ret))
            {
                return services::throwIfPossible(services::ErrorODBC);
            }
        }

        if (_modifiersManager)
        {
            DAAL_CHECK_STATUS(status, _modifiersManager->prepare(featuresInfo, *_fetchBuffer));
        }

        return status;
    }

    services::Status fillDictionary(DataSourceDictionary & dictionary, const internal::SQLFeaturesInfo & featuresInfo)
    {
        if (_modifiersManager)
        {
            return _modifiersManager->fillDictionary(dictionary);
        }

        const size_t nFeatures  = featuresInfo.getNumberOfFeatures();
        services::Status status = dictionary.setNumberOfFeatures(nFeatures);
        if (!status)
        {
            return services::throwIfPossible(status);
        }

        for (size_t i = 0; i < nFeatures; i++)
        {
            dictionary[i].ntFeature.setType<DAAL_DATA_TYPE>();
            dictionary[i].ntFeature.featureType = features::DAAL_CONTINUOUS;
        }

        return status;
    }

private:
    internal::SQLFetchBufferPtr _fetchBuffer;
    BlockDescriptor<DAAL_DATA_TYPE> _currentRowBlock;
    services::SharedPtr<services::ErrorCollection> _errors;
    modifiers::sql::internal::ModifiersManagerPtr _modifiersManager;
};

typedef SQLFeatureManager MySQLFeatureManager;

/** @} */
} // namespace interface1

using interface1::SQLFeatureManager;
using interface1::MySQLFeatureManager;

} // namespace data_management
} // namespace daal

#endif
