/* file: data_source.h */
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
//  Declaration and implementation of the base data source class.
//--
*/

#ifndef __DATA_SOURCE_H__
#define __DATA_SOURCE_H__

#include "data_management/data/data_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/aos_numeric_table.h"
#include "data_management/data/soa_numeric_table.h"

#include "data_management/data_source/data_source_utils.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCEIFACE"></a>
 *  \brief Abstract interface class that defines the interface for a data management component responsible for
 *  representation of data in the raw format. This class declares the most generic methods for data access.
 */
class DataSourceIface
{
public:
    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__DATASOURCESTATUS"></a>
     * \brief Specifies the status of the Data Source
     */
    enum DataSourceStatus
    {
        readyForLoad   = 1, /*!< Data is ready to be loaded via loadDataBlock() function */
        waitingForRows = 2, /*!< No data is available, but it may be ready in future */
        endOfData      = 3, /*!< No data is available */
        notReady       = 4  /*!< DataSource not ready for loading */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__DICTIONARYCREATIONFLAG"></a>
     * \brief Specifies whether a Data %Dictionary is created from the context of a Data Source
     */
    enum DictionaryCreationFlag
    {
        notDictionaryFromContext = 1, /*!< Do not create dictionary automatically */
        doDictionaryFromContext  = 2  /*!< Do create dictionary when needed */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__NUMERICTABLEALLOCATIONFLAG"></a>
     * \brief Specifies whether a Numeric Table is allocated inside of the Data Source object
     */
    enum NumericTableAllocationFlag
    {
        notAllocateNumericTable = 1, /*!< Do not allocate Numeric Table automatically */
        doAllocateNumericTable  = 2  /*!< Do allocate Numeric Table when needed */
    };

public:
    /**
     *  Returns a pointer to a data dictionary
     *  \return Pointer to the Data %Dictionary
     */
    DAAL_DEPRECATED_VIRTUAL virtual DataSourceDictionary * getDictionary() = 0;

    /**
     *  Returns a shared pointer to a data dictionary
     *  \return Shared pointer to the Data %Dictionary
     */
    virtual DataSourceDictionaryPtr getDictionarySharedPtr() = 0;

    /**
     *  Sets a predefined Data %Dictionary
     */
    virtual services::Status setDictionary(DataSourceDictionary * dict) = 0;

    /**
     *  Creates a Data Dictionary by extracting information from a Data Source
     */
    virtual services::Status createDictionaryFromContext() = 0;

    /**
     *  Returns the status of a Data Source
     *  \return Status of the Data Source
     */
    virtual DataSourceStatus getStatus() = 0;

    /**
     *  Returns the number of columns in a Data Source
     *  \return Number of columns
     */
    virtual size_t getNumberOfColumns() = 0;

    /**
     *  Returns the number of columns in a Numeric Table associated with a Data Source
     *  \return Number of columns
     */
    virtual size_t getNumericTableNumberOfColumns() = 0;

    /**
     *  Returns the number of rows available in a Data Source
     *  \return Number of rows
     */
    virtual size_t getNumberOfAvailableRows() = 0;

    /**
     *  Allocates a Numeric Table associated with a Data Source
     */
    virtual services::Status allocateNumericTable() = 0;

    /**
     *  Returns a pointer to a Numeric Table associated with a Data Source
     *  \return Pointer to the Numeric Table
     */
    virtual NumericTablePtr getNumericTable() = 0;

    /**
     *  Returns a pointer to a Numeric Table associated with a Data Source
     */
    virtual void freeNumericTable() = 0;

    /**
     *  Loads a data block of a specified size into an internally allocated Numeric Table
     *  \param[in] maxRows Maximum number of rows to load from a Data Source into the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows) = 0;

    /**
     *  Loads a data block of a specified size into an internally allocated Numeric Table
     *  \param[in] maxRows   Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param[in] rowOffset Write data starting from rowOffset row
     *  \param[in] fullRows  Maximum number of rows to allocate in the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows) = 0;

    /**
     *  Loads a data block of a specified size into a provided Numeric Table
     *  \param[in] maxRows Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param[in] nt      Pointer to the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows, NumericTable * nt) = 0;

    /**
     *  Loads a data block of a specified size into an internally allocated Numeric Table
     *  \param[in] maxRows   Maximum number of rows to load from a Data Source into the Numeric Table
     *  \param[in] rowOffset Write data starting from rowOffset row
     *  \param[in] fullRows  Maximum number of rows to allocate in the Numeric Table
     *  \param[in] nt        Pointer to the Numeric Table
     */
    virtual size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows, NumericTable * nt) = 0;

    /**
     *  Loads a data block into an internally allocated Numeric Table
     */
    virtual size_t loadDataBlock() = 0;

    /**
     *  Loads a data block into a provided Numeric Table
     *  \param[in] nt      Pointer to the Numeric Table
     */
    virtual size_t loadDataBlock(NumericTable * nt) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCE"></a>
 *  \brief Implements the abstract DataSourceIface interface
 */
class DataSource : public DataSourceIface
{
public:
    DataSource()
        : _dict(),
          _autoNumericTableFlag(doAllocateNumericTable),
          _autoDictionaryFlag(doDictionaryFromContext),
          _errors(new services::ErrorCollection()),
          _initialMaxRows(10)
    {}

    virtual ~DataSource() {}

    DAAL_DEPRECATED_VIRTUAL DataSourceDictionary * getDictionary() DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if (!s) return NULL;
        return _dict.get();
    }

    DataSourceDictionaryPtr getDictionarySharedPtr() DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if (!s) return DataSourceDictionaryPtr();
        return _dict;
    }

    services::Status setDictionary(DataSourceDictionary * dict) DAAL_C11_OVERRIDE
    {
        if (_dict) return services::throwIfPossible(services::Status(services::ErrorDictionaryAlreadyAvailable));

        services::Status s = dict->checkDictionary();
        if (!s) return services::throwIfPossible(s);

        _dict.reset(dict, services::EmptyDeleter());
        return services::Status();
    }

    services::Status createDictionaryFromContext() DAAL_C11_OVERRIDE
    {
        return services::throwIfPossible(services::Status(services::ErrorMethodNotSupported));
    }

    size_t loadDataBlock(size_t maxRows) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        s.add(checkNumericTable());
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        return loadDataBlock(maxRows, this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(size_t /*maxRows*/, NumericTable * /*nt*/) DAAL_C11_OVERRIDE
    {
        this->_status.add(services::throwIfPossible(services::ErrorMethodNotSupported));
        return 0;
    }

    size_t loadDataBlock(size_t maxRows, size_t rowOffset, size_t fullRows) DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if (s) s.add(checkNumericTable());
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        return loadDataBlock(maxRows, rowOffset, fullRows, this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(size_t /*maxRows*/, size_t /*rowOffset*/, size_t /*fullRows*/, NumericTable * /*nt*/) DAAL_C11_OVERRIDE
    {
        this->_status.add(services::throwIfPossible(services::ErrorMethodNotSupported));
        return 0;
    }

    size_t loadDataBlock() DAAL_C11_OVERRIDE
    {
        services::Status s = checkDictionary();
        if (s)
        {
            s.add(checkNumericTable());
        }
        if (!s)
        {
            this->_status.add(services::throwIfPossible(s));
            return 0;
        }
        return loadDataBlock(this->DataSource::_spnt.get());
    }

    size_t loadDataBlock(NumericTable * /*nt*/) DAAL_C11_OVERRIDE
    {
        this->_status.add(services::throwIfPossible(services::ErrorMethodNotSupported));
        return 0;
    }

    NumericTablePtr getNumericTable() DAAL_C11_OVERRIDE
    {
        checkNumericTable();
        return _spnt;
    }

    size_t getNumberOfColumns() DAAL_C11_OVERRIDE
    {
        checkDictionary();
        return _dict ? _dict->getNumberOfFeatures() : 0;
    }

    /**
     * Returns errors during the computation
     * \return Errors during the computation
     */
    services::Status status() const
    {
        services::Status s = _status;
        /*if(_spnt.get())
            s.add(_spnt->status());*/
        return s;
    }

    /**
    *  For backward compatibility. Returns errors stored on the object
    *  \return Errors stored on the object
    *  \DAAL_DEPRECATED
    */
    DAAL_DEPRECATED services::SharedPtr<services::ErrorCollection> getErrors() { return status().getCollection(); }

    virtual size_t getNumericTableNumberOfColumns() DAAL_C11_OVERRIDE { return getNumberOfColumns(); }

protected:
    DataSourceDictionaryPtr _dict;
    NumericTablePtr _spnt;

    NumericTableAllocationFlag _autoNumericTableFlag;
    DictionaryCreationFlag _autoDictionaryFlag;
    services::Status _status;
    services::SharedPtr<services::ErrorCollection> _errors;
    size_t _initialMaxRows;

    /**
     * Checks a Numeric Table
     */
    services::Status checkNumericTable()
    {
        if (_spnt.get() == NULL)
        {
            if (_autoNumericTableFlag == notAllocateNumericTable)
                return services::throwIfPossible(services::Status(services::ErrorNumericTableNotAllocated));
            return allocateNumericTable();
        }
        return services::Status();
    }

    /**
     * Checks a Data Dictionary
     */
    services::Status checkDictionary()
    {
        if (_dict == 0)
        {
            if (_autoDictionaryFlag == notDictionaryFromContext)
                return services::throwIfPossible(services::Status(services::ErrorDictionaryNotAvailable));
            return createDictionaryFromContext();
        }
        return services::Status();
    }

    /**
     *  Allocates a Numeric Table that corresponds to the template type
     *
     *  \tparam NumericTableType - Numeric Table type.
     *
     *  \param   nt      - Pointer to the allocated Numeric Table
     *
     *  \return          - Allocation status: True if the table is allocated, false otherwise.
     */
    template <typename NumericTableType>
    services::Status allocateNumericTableImpl(services::SharedPtr<NumericTableType> & nt);

    /**
     *  Allocates a homogeneous Numeric Table that corresponds to the template type
     *
     *  \tparam FPType - Type of the homogeneous Numeric Table
     *
     *  \param   nt      - Pointer to the allocated Numeric Table
     *
     *  \return          - Allocation status: True if the table is allocated, false otherwise.
     */
    template <typename FPType>
    services::Status allocateNumericTableImpl(services::SharedPtr<HomogenNumericTable<FPType> > & nt);

    size_t getStructureSize()
    {
        size_t structureSize = 0;
        size_t nFeatures     = _dict->getNumberOfFeatures();
        for (size_t i = 0; i < nFeatures; i++)
        {
            structureSize += (*_dict)[i].ntFeature.typeSize;
        }
        return structureSize;
    }

    virtual services::Status setNumericTableDictionary(NumericTablePtr nt)
    {
        if (!nt) return services::throwIfPossible(services::Status(services::ErrorNullNumericTable));
        NumericTableDictionaryPtr ntDict = nt->getDictionarySharedPtr();
        if (!ntDict) return services::throwIfPossible(services::Status(services::ErrorDictionaryNotAvailable));

        size_t nFeatures = ntDict->getNumberOfFeatures();

        for (size_t i = 0; i < nFeatures; i++)
        {
            (*ntDict)[i] = (*_dict)[i].ntFeature;
        }
        return services::Status();
    }
};

template <typename NumericTableType>
inline services::Status DataSource::allocateNumericTableImpl(services::SharedPtr<NumericTableType> & nt)
{
    nt = services::SharedPtr<NumericTableType>();
    return services::Status();
}

template <>
inline services::Status DataSource::allocateNumericTableImpl(AOSNumericTablePtr & nt)
{
    size_t nFeatures     = _dict->getNumberOfFeatures();
    size_t structureSize = getStructureSize();
    services::Status s;
    nt = AOSNumericTable::create(structureSize, nFeatures, 0, &s);
    if (!s) return s;
    s |= setNumericTableDictionary(nt);
    return s;
}

template <>
inline services::Status DataSource::allocateNumericTableImpl(SOANumericTablePtr & nt)
{
    nt = SOANumericTablePtr();
    return services::Status();
}

template <typename FPType>
inline services::Status DataSource::allocateNumericTableImpl(services::SharedPtr<HomogenNumericTable<FPType> > & nt)
{
    size_t nFeatures = getNumericTableNumberOfColumns();
    services::Status s;
    nt = HomogenNumericTable<FPType>::create(nFeatures, 0, NumericTableIface::doNotAllocate, &s);
    if (!s) return s;
    s |= setNumericTableDictionary(nt);
    return s;
}

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATASOURCETEMPLATE"></a>
 *  \brief Implements the abstract DataSourceIface interface
 */
template <typename _numericTableType, typename _summaryStatisticsType = DAAL_SUMMARY_STATISTICS_TYPE>
class DataSourceTemplate : public DataSource
{
public:
    typedef _numericTableType numericTableType;

public:
    DataSourceTemplate(NumericTableAllocationFlag doAllocateNumericTable, DictionaryCreationFlag doCreateDictionaryFromContext) : DataSource()
    {
        DataSource::_autoNumericTableFlag = doAllocateNumericTable;
        DataSource::_autoDictionaryFlag   = doCreateDictionaryFromContext;
    }

    virtual ~DataSourceTemplate() {}

    virtual void freeNumericTable() DAAL_C11_OVERRIDE { _spnt = NumericTablePtr(); }

    virtual services::Status allocateNumericTable() DAAL_C11_OVERRIDE
    {
        if (_spnt.get() != NULL) return services::throwIfPossible(services::Status(services::ErrorNumericTableAlreadyAllocated));

        services::Status s = checkDictionary();
        if (!s) return s;

        services::SharedPtr<numericTableType> nt;

        s |= allocateNumericTableImpl(nt);
        _spnt = nt;

        services::SharedPtr<HomogenNumericTable<_summaryStatisticsType> > ssNt;

        s |= allocateNumericTableImpl(ssNt);
        _spnt->basicStatistics.set(NumericTable::minimum, ssNt);

        s |= allocateNumericTableImpl(ssNt);
        _spnt->basicStatistics.set(NumericTable::maximum, ssNt);

        s |= allocateNumericTableImpl(ssNt);
        _spnt->basicStatistics.set(NumericTable::sum, ssNt);

        s |= allocateNumericTableImpl(ssNt);
        _spnt->basicStatistics.set(NumericTable::sumSquares, ssNt);
        return s;
    }

protected:
    services::Status resizeNumericTableImpl(const size_t linesToLoad, NumericTable * nt)
    {
        if (!nt) return services::Status(services::ErrorNullInputNumericTable);

        if (!_dict) return services::Status(services::ErrorDictionaryNotAvailable);

        size_t nFeatures = getNumericTableNumberOfColumns();

        if (nt->getNumberOfColumns() < nFeatures)
        {
            nt->getDictionarySharedPtr()->setNumberOfFeatures(nFeatures);
        }

        nt->resize(0);
        nt->resize(linesToLoad);

        const size_t nCols = nt->getNumberOfColumns();

        nt->allocateBasicStatistics();

        NumericTablePtr ntMin   = nt->basicStatistics.get(NumericTable::minimum);
        NumericTablePtr ntMax   = nt->basicStatistics.get(NumericTable::maximum);
        NumericTablePtr ntSum   = nt->basicStatistics.get(NumericTable::sum);
        NumericTablePtr ntSumSq = nt->basicStatistics.get(NumericTable::sumSquares);

        if (ntMin->getNumberOfColumns() != nCols || ntMin->getNumberOfRows() != 1)
        {
            if (ntMin->getNumberOfColumns() != nCols)
            {
                ntMin->getDictionarySharedPtr()->setNumberOfFeatures(nCols);
            }
            ntMin->resize(1);
        }

        if (ntMax->getNumberOfColumns() != nCols || ntMax->getNumberOfRows() != 1)
        {
            if (ntMax->getNumberOfColumns() != nCols)
            {
                ntMax->getDictionarySharedPtr()->setNumberOfFeatures(nCols);
            }
            ntMax->resize(1);
        }

        if (ntSum->getNumberOfColumns() != nCols || ntSum->getNumberOfRows() != 1)
        {
            if (ntSum->getNumberOfColumns() != nCols)
            {
                ntSum->getDictionarySharedPtr()->setNumberOfFeatures(nCols);
            }
            ntSum->resize(1);
        }

        if (ntSumSq->getNumberOfColumns() != nCols || ntSumSq->getNumberOfRows() != 1)
        {
            if (ntSumSq->getNumberOfColumns() != nCols)
            {
                ntSumSq->getDictionarySharedPtr()->setNumberOfFeatures(nCols);
            }
            ntSumSq->resize(1);
        }
        return services::Status();
    }

    services::Status updateStatistics(size_t ntRowIndex, NumericTable * nt, DAAL_DATA_TYPE * row, size_t offset = 0)
    {
        if (!nt) return services::Status(services::ErrorNullInputNumericTable);
        if (!row) return services::Status(services::ErrorNullPtr);

        NumericTablePtr ntMin   = nt->basicStatistics.get(NumericTable::minimum);
        NumericTablePtr ntMax   = nt->basicStatistics.get(NumericTable::maximum);
        NumericTablePtr ntSum   = nt->basicStatistics.get(NumericTable::sum);
        NumericTablePtr ntSumSq = nt->basicStatistics.get(NumericTable::sumSquares);

        BlockDescriptor<_summaryStatisticsType> blockMin;
        BlockDescriptor<_summaryStatisticsType> blockMax;
        BlockDescriptor<_summaryStatisticsType> blockSum;
        BlockDescriptor<_summaryStatisticsType> blockSumSq;

        ntMin->getBlockOfRows(0, 1, readWrite, blockMin);
        ntMax->getBlockOfRows(0, 1, readWrite, blockMax);
        ntSum->getBlockOfRows(0, 1, readWrite, blockSum);
        ntSumSq->getBlockOfRows(0, 1, readWrite, blockSumSq);

        _summaryStatisticsType * minimum    = blockMin.getBlockPtr();
        _summaryStatisticsType * maximum    = blockMax.getBlockPtr();
        _summaryStatisticsType * sum        = blockSum.getBlockPtr();
        _summaryStatisticsType * sumSquares = blockSumSq.getBlockPtr();

        size_t nCols = nt->getNumberOfColumns();

        if (minimum == NULL || maximum == NULL || sum == NULL || sumSquares == NULL)
        {
            ntMin->releaseBlockOfRows(blockMin);
            ntMax->releaseBlockOfRows(blockMax);
            ntSum->releaseBlockOfRows(blockSum);
            ntSumSq->releaseBlockOfRows(blockSumSq);
            return services::Status(services::ErrorIncorrectInputNumericTable);
        }

        row += (ntRowIndex + offset) * nt->getNumberOfColumns();

        if (ntRowIndex != 0)
        {
            for (size_t i = 0; i < nCols; i++)
            {
                if (minimum[i] > row[i])
                {
                    minimum[i] = row[i];
                }
                if (maximum[i] < row[i])
                {
                    maximum[i] = row[i];
                }
                sum[i] += row[i];
                sumSquares[i] += row[i] * row[i];
            }
        }
        else
        {
            for (size_t i = 0; i < nCols; i++)
            {
                minimum[i]    = row[i];
                maximum[i]    = row[i];
                sum[i]        = row[i];
                sumSquares[i] = row[i] * row[i];
            }
        }

        ntMin->releaseBlockOfRows(blockMin);
        ntMax->releaseBlockOfRows(blockMax);
        ntSum->releaseBlockOfRows(blockSum);
        ntSumSq->releaseBlockOfRows(blockSumSq);
        return services::Status();
    }

    services::Status combineSingleStatistics(NumericTable * ntSrc, NumericTable * ntDst, bool wasEmpty, NumericTable::BasicStatisticsId id)
    {
        if (ntSrc == NULL || ntDst == NULL) return services::Status(services::ErrorNullInputNumericTable);

        NumericTablePtr ntSrcStat = ntSrc->basicStatistics.get(id);
        NumericTablePtr ntDstStat = ntDst->basicStatistics.get(id);

        BlockDescriptor<_summaryStatisticsType> blockSrc;
        BlockDescriptor<_summaryStatisticsType> blockDst;

        ntSrcStat->getBlockOfRows(0, 1, readOnly, blockSrc);
        ntDstStat->getBlockOfRows(0, 1, readWrite, blockDst);

        const _summaryStatisticsType * src = blockSrc.getBlockPtr();
        _summaryStatisticsType * dst       = blockDst.getBlockPtr();

        if (src == NULL || dst == NULL)
        {
            ntSrcStat->releaseBlockOfRows(blockSrc);
            ntDstStat->releaseBlockOfRows(blockDst);
            return services::Status(services::ErrorIncorrectInputNumericTable);
        }

        const size_t nColsSrc = ntSrc->getNumberOfColumns();
        const size_t nCols    = ntDst->getNumberOfColumns();

        if (nCols != nColsSrc)
        {
            ntSrcStat->releaseBlockOfRows(blockSrc);
            ntDstStat->releaseBlockOfRows(blockDst);
            return services::Status(services::ErrorIncorrectInputNumericTable);
        }

        if (wasEmpty)
        {
            for (size_t i = 0; i < nCols; i++)
            {
                dst[i] = src[i];
            }
        }
        else
        {
            if (id == NumericTable::minimum)
            {
                for (size_t i = 0; i < nCols; i++)
                {
                    if (dst[i] > src[i])
                    {
                        dst[i] = src[i];
                    }
                }
            }
            else if (id == NumericTable::maximum)
            {
                for (size_t i = 0; i < nCols; i++)
                {
                    if (dst[i] < src[i])
                    {
                        dst[i] = src[i];
                    }
                }
            }
            else if (id == NumericTable::sum)
            {
                for (size_t i = 0; i < nCols; i++)
                {
                    dst[i] += src[i];
                }
            }
            else if (id == NumericTable::sumSquares)
            {
                for (size_t i = 0; i < nCols; i++)
                {
                    dst[i] += src[i];
                }
            }
        }

        ntSrcStat->releaseBlockOfRows(blockSrc);
        ntDstStat->releaseBlockOfRows(blockDst);
        return services::Status();
    }

    services::Status combineStatistics(NumericTable * ntSrc, NumericTable * ntDst, bool wasEmpty)
    {
        services::Status s;
        s.add(combineSingleStatistics(ntSrc, ntDst, wasEmpty, NumericTable::minimum));
        s.add(combineSingleStatistics(ntSrc, ntDst, wasEmpty, NumericTable::maximum));
        s.add(combineSingleStatistics(ntSrc, ntDst, wasEmpty, NumericTable::sum));
        s.add(combineSingleStatistics(ntSrc, ntDst, wasEmpty, NumericTable::sumSquares));
        return s;
    }
};

/** @} */
} // namespace interface1

using interface1::DataSourceIface;
using interface1::DataSource;
using interface1::DataSourceTemplate;

} // namespace data_management
} // namespace daal

#endif
