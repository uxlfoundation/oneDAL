/* file: kdb_feature_manager.h */
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
//  Implementation of the KDB data source class.
//--
*/
#ifndef __KDB_FEATURE_MANAGER_H__
#define __KDB_FEATURE_MANAGER_H__

#include "services/daal_memory.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

#include <k.h>

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-KDBFEATUREMANAGER"></a>
 * \brief Contains KDB-specific commands
 * \DAAL_DEPRECATED
 */
class KDBFeatureManager
{
public:
    DAAL_DEPRECATED KDBFeatureManager() : _errors(new services::ErrorCollection()) {}

    /**
     *  Creates a data dictionary from a KDB table
     *
     *  \param[in]   table Table from a KDB query
     *  \param[out]  dict  Dictionary to be created
     */
    void createDictionaryFromTable(const K & table, DataSourceDictionary * dict)
    {
        K featureNames   = kK(table)[0];
        K featureData    = kK(table)[1];
        size_t nFeatures = featureNames->n;

        dict->setNumberOfFeatures(nFeatures);

        for (size_t i = 0; i < nFeatures; i++)
        {
            DataSourceFeature & feature = (*dict)[i];

            feature.setFeatureName(kS(featureNames)[i]);

            K column = kK(featureData)[i];

            switch (column->t)
            {
            case (KF):
            case (KZ): feature.setType<double>(); break;
            case (KE): feature.setType<float>(); break;
            case (KB):
            case (KG):
            case (KC): feature.setType<char>(); break;
            case (KH): feature.setType<short>(); break;
            case (KI):
            case (KM):
            case (KD):
            case (KU):
            case (KV):
            case (KT): feature.setType<int>(); break;
            case (KJ): feature.setType<DAAL_INT64>(); break;
            default: _errors->add(services::ErrorKDBTypeUnsupported); break;
            }
        }
    }

    /**
     *  Creates a data dictionary from a KDB list
     *
     *  \param[in]   lst   List from a KDB query
     *  \param[out]  dict  Dictionary to be created
     */
    void createDictionaryFromList(const K & lst, DataSourceDictionary * dict)
    {
        size_t nFeatures = lst->n;

        dict->setNumberOfFeatures(nFeatures);

        for (size_t i = 0; i < nFeatures; i++)
        {
            DataSourceFeature & feature = (*dict)[i];

            I curType;

            if (lst->t == 0)
                curType = kK(lst)[i]->t;
            else
                curType = lst->t;

            if (curType < 0) curType = -curType;

            switch (curType)
            {
            case (KF):
            case (KZ): feature.setType<double>(); break;
            case (KE): feature.setType<float>(); break;
            case (KB):
            case (KG):
            case (KC): feature.setType<char>(); break;
            case (KH): feature.setType<short>(); break;
            case (KI):
            case (KM):
            case (KD):
            case (KU):
            case (KV):
            case (KT): feature.setType<int>(); break;
            case (KJ): feature.setType<DAAL_INT64>(); break;
            default: _errors->add(services::ErrorKDBTypeUnsupported); break;
            }
        }
    }

    void statementResultsNumericTableFromColumnData(const K & columnData, NumericTable * nt, size_t nRows)
    {
        BlockDescriptor<DAAL_DATA_TYPE> block;
        nt->getBlockOfRows(0, nRows, writeOnly, block);
        DAAL_DATA_TYPE * blockPtr = block.getBlockPtr();
        size_t nFeatures          = nt->getNumberOfColumns();
        for (size_t row = 0; row < nRows; row++)
        {
            for (size_t col = 0; col < nFeatures; col++)
            {
                K column = kK(columnData)[col];
                switch (column->t)
                {
                case (KB): blockPtr[row * nFeatures + col] = kG(column)[row]; break;
                case (KG): blockPtr[row * nFeatures + col] = kG(column)[row]; break;
                case (KH): blockPtr[row * nFeatures + col] = kH(column)[row]; break;
                case (KI):
                case (KM):
                case (KD):
                case (KU):
                case (KV):
                case (KT): blockPtr[row * nFeatures + col] = kI(column)[row]; break;
                case (KJ):
                case (KP):
                case (KN): blockPtr[row * nFeatures + col] = kJ(column)[row]; break;
                case (KE): blockPtr[row * nFeatures + col] = kE(column)[row]; break;
                case (KF): blockPtr[row * nFeatures + col] = kF(column)[row]; break;
                default: _errors->add(services::ErrorKDBTypeUnsupported); break;
                }
            }
        }
        nt->releaseBlockOfRows(block);
    }

    void statementResultsNumericTableFromList(const K & lst, NumericTable * nt, size_t nRows)
    {
        BlockDescriptor<DAAL_DATA_TYPE> block;
        nt->getBlockOfRows(0, nRows, writeOnly, block);
        DAAL_DATA_TYPE * blockPtr = block.getBlockPtr();
        size_t nFeatures          = nt->getNumberOfColumns();
        for (size_t row = 0; row < nRows; row++)
        {
            K data;
            if (lst->t == 0)
                data = kK(lst)[row];
            else
                data = lst;
            for (size_t col = 0; col < nFeatures; col++)
            {
                switch (data->t)
                {
                case (KB): blockPtr[row * nFeatures + col] = kG(data)[col]; break;
                case (KG): blockPtr[row * nFeatures + col] = kG(data)[col]; break;
                case (KH): blockPtr[row * nFeatures + col] = kH(data)[col]; break;
                case (KI):
                case (KM):
                case (KD):
                case (KU):
                case (KV):
                case (KT): blockPtr[row * nFeatures + col] = kI(data)[col]; break;
                case (KJ):
                case (KP):
                case (KN): blockPtr[row * nFeatures + col] = kJ(data)[col]; break;
                case (KE): blockPtr[row * nFeatures + col] = kE(data)[col]; break;
                case (KF): blockPtr[row * nFeatures + col] = kF(data)[col]; break;
                default: _errors->add(services::ErrorKDBTypeUnsupported); break;
                }
            }
            if (lst->t == 0) r0(data);
        }
        nt->releaseBlockOfRows(block);
    }

    services::SharedPtr<services::ErrorCollection> getErrors() { return _errors; }

private:
    services::SharedPtr<services::ErrorCollection> _errors;
};

} // namespace interface1
using interface1::KDBFeatureManager;

} // namespace data_management
} // namespace daal
#endif
