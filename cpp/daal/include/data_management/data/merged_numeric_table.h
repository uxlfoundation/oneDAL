/* file: merged_numeric_table.h */
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
//  Implementation of merge numeric table.
//--
*/

#ifndef __MERGED_NUMERIC_TABLE_H__
#define __MERGED_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"
#include "services/daal_memory.h"
#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__MERGEDNUMERICTABLE"></a>
 *  \brief Class that provides methods to access a collection of numeric tables as if they are joined by columns
 */
class DAAL_EXPORT MergedNumericTable : public NumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    /**
     *  Constructor for an empty merge Numeric Table
     *  \DAAL_DEPRECATED_USE{ MergedNumericTable::create }
     */
    MergedNumericTable();

    /**
     *  Constructor for a merge Numeric Table consisting of one table
     *  \param[in]  table       Pointer to the table
     *  \DAAL_DEPRECATED_USE{ MergedNumericTable::create }
     */
    MergedNumericTable(NumericTablePtr table);

    /**
     *  Constructor for a merge Numeric Table consisting of two tables
     *  \param[in]  first      Pointer to the first table
     *  \param[in]  second     Pointer to the second table
     *  \DAAL_DEPRECATED_USE{ MergedNumericTable::create }
     */
    MergedNumericTable(NumericTablePtr first, NumericTablePtr second);

    /**
     * Constructor for an empty merge Numeric Table
     * \param[out] stat  Status of the MergedNumericTable construction
     */
    static services::SharedPtr<MergedNumericTable> create(services::Status * stat = NULL);

    /**
     * Constructs a merge Numeric Table consisting of one table
     * \param[in]  nestedTable  Pointer to the table
     * \param[out] stat         Status of the MergedNumericTable construction
     * \return     Merge Numeric Table of one table
     */
    static services::SharedPtr<MergedNumericTable> create(const NumericTablePtr & nestedTable, services::Status * stat = NULL);

    /**
     * Constructs a merge Numeric Table consisting of two tables
     * \param[in]  first   Pointer to the first table
     * \param[in]  second  Pointer to the second table
     * \param[out] stat    Status of the MergedNumericTable construction
     * \return     Merge Numeric Table of two tables
     */
    static services::SharedPtr<MergedNumericTable> create(const NumericTablePtr & first, const NumericTablePtr & second,
                                                          services::Status * stat = NULL);

    /**
     *  Adds the table to the right of the merge Numeric Table
     *  \param[in] table Pointer to the table
     */
    services::Status addNumericTable(NumericTablePtr table)
    {
        if (table->getDataLayout() & csrArray) return services::Status(services::ErrorIncorrectTypeOfInputNumericTable);

        _tables->push_back(table);

        size_t ncols = getNumberOfColumns();
        size_t cols  = table->getNumberOfColumns();

        services::Status s;
        DAAL_CHECK_STATUS(s, setNumberOfColumnsImpl(ncols + cols));

        for (size_t i = 0; i < cols; i++)
        {
            NumericTableFeature & f = table->getDictionarySharedPtr()->operator[](i);
            _ddict->setFeature(f, ncols + i);
        }

        size_t obs = table->getNumberOfRows();
        if (obs != _obsnum)
        {
            if (obs < _obsnum || _tables->size() == 1)
            {
                _obsnum = obs;
            }
            DAAL_CHECK_STATUS(s, setNumberOfRowsImpl(_obsnum));
        }
        return s;
    }

    //the descriptions of the methods below are inherited from the base class
    services::Status resize(size_t nrow) DAAL_C11_OVERRIDE
    {
        for (size_t i = 0; i < _tables->size(); i++)
        {
            NumericTable * nt  = (NumericTable *)(_tables->operator[](i).get());
            services::Status s = nt->resize(nrow);
            if (!s) return s;
        }
        _obsnum = nrow;
        return services::Status();
    }

    MemoryStatus getDataMemoryStatus() const DAAL_C11_OVERRIDE
    {
        if (_tables->size() == 0)
        {
            return notAllocated;
        }

        for (size_t i = 0; i < _tables->size(); i++)
        {
            NumericTable * nt = (NumericTable *)(_tables->operator[](i).get());
            if (nt->getDataMemoryStatus() == notAllocated)
            {
                return notAllocated;
            }
        }

        return internallyAllocated;
    }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }

    services::Status allocateBasicStatistics() DAAL_C11_OVERRIDE
    {
        size_t ncols   = getNumberOfColumns();
        size_t ntables = _tables->size();
        services::SharedPtr<MergedNumericTable> minMergeNT(new MergedNumericTable());
        services::SharedPtr<MergedNumericTable> maxMergeNT(new MergedNumericTable());
        services::SharedPtr<MergedNumericTable> sumMergeNT(new MergedNumericTable());
        services::SharedPtr<MergedNumericTable> sumSqMergeNT(new MergedNumericTable());
        for (size_t i = 0; i < ntables; i++)
        {
            NumericTable * nt = (NumericTable *)(_tables->operator[](i).get());
            nt->allocateBasicStatistics();
            minMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::minimum));
            maxMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::maximum));
            sumMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::sum));
            sumSqMergeNT->addNumericTable(nt->basicStatistics.get(NumericTable::sumSquares));
        }
        if (basicStatistics.get(NumericTable::minimum).get() == NULL || basicStatistics.get(NumericTable::minimum)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::minimum, minMergeNT);
        }
        if (basicStatistics.get(NumericTable::maximum).get() == NULL || basicStatistics.get(NumericTable::maximum)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::maximum, maxMergeNT);
        }
        if (basicStatistics.get(NumericTable::sum).get() == NULL || basicStatistics.get(NumericTable::sum)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::sum, sumMergeNT);
        }
        if (basicStatistics.get(NumericTable::sumSquares).get() == NULL
            || basicStatistics.get(NumericTable::sumSquares)->getNumberOfColumns() != ncols)
        {
            basicStatistics.set(NumericTable::sumSquares, sumSqMergeNT);
        }
        return services::Status();
    }

protected:
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        arch->setSharedPtrObj(_tables);

        return services::Status();
    }

private:
    template <typename T>
    void internal_inner_repack(size_t pos, size_t cols, size_t rows, size_t ncols, T * src, T * dst)
    {
        size_t i, j;

        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                dst[i * ncols + j + pos] = src[i * cols + j];
            }
        }
    }

    template <typename T>
    void internal_outer_repack(size_t pos, size_t cols, size_t rows, size_t ncols, T * src, T * dst)
    {
        size_t i, j;

        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                dst[i * cols + j] = src[i * ncols + j + pos];
            }
        }
    }

protected:
    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        services::Status s;
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (rwFlag & (int)readOnly)
        {
            size_t cols = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lcols      = nt->getNumberOfColumns();

                s |= nt->getBlockOfRows(idx, nrows, readOnly, innerBlock);

                internal_inner_repack<T>(cols, lcols, nrows, ncols, innerBlock.getBlockPtr(), block.getBlockPtr());

                s |= nt->releaseBlockOfRows(innerBlock);

                cols += lcols;
            }
        }
        return s;
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        services::Status s;
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols  = getNumberOfColumns();
            size_t nrows  = block.getNumberOfRows();
            size_t offset = block.getRowsOffset();
            size_t cols   = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lcols      = nt->getNumberOfColumns();

                s |= nt->getBlockOfRows(offset, nrows, writeOnly, innerBlock);

                internal_outer_repack<T>(cols, lcols, nrows, ncols, block.getBlockPtr(), innerBlock.getBlockPtr());

                s |= nt->releaseBlockOfRows(innerBlock);

                cols += lcols;
            }
        }
        block.reset();
        return s;
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        services::Status s;
        size_t nobs = getNumberOfRows();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (rwFlag & (int)readOnly)
        {
            T * buffer = block.getBlockPtr();
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lcols      = nt->getNumberOfColumns();

                if (lcols > feat_idx)
                {
                    BlockDescriptor<T> innerBlock;
                    s |= nt->getBlockOfColumnValues(feat_idx, idx, nrows, readOnly, innerBlock);
                    T * location = innerBlock.getBlockPtr();
                    for (size_t i = 0; i < nrows; i++)
                    {
                        buffer[i] = location[i];
                    }
                    s |= nt->releaseBlockOfColumnValues(innerBlock);
                    break;
                }

                feat_idx -= lcols;
            }
        }
        return s;
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        services::Status s;
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();
            size_t idx      = block.getRowsOffset();
            size_t nrows    = block.getNumberOfRows();
            T * buffer      = block.getBlockPtr();
            for (size_t k = 0; k < _tables->size(); k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lcols      = nt->getNumberOfColumns();

                if (lcols > feat_idx)
                {
                    BlockDescriptor<T> innerBlock;
                    s |= nt->getBlockOfColumnValues(feat_idx, idx, nrows, writeOnly, innerBlock);
                    T * location = innerBlock.getBlockPtr();
                    for (size_t i = 0; i < nrows; i++)
                    {
                        location[i] = buffer[i];
                    }
                    s |= nt->releaseBlockOfColumnValues(innerBlock);
                    break;
                }

                feat_idx -= lcols;
            }
        }
        block.reset();
        return s;
    }

    services::Status setNumberOfRowsImpl(size_t nrow) DAAL_C11_OVERRIDE;

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE;

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE;

protected:
    DataCollectionPtr _tables;

    MergedNumericTable(services::Status & st);

    MergedNumericTable(const NumericTablePtr & table, services::Status & st);

    MergedNumericTable(const NumericTablePtr & first, const NumericTablePtr & second, services::Status & st);
};
typedef services::SharedPtr<MergedNumericTable> MergedNumericTablePtr;
/** @} */
} // namespace interface1
using interface1::MergedNumericTable;
using interface1::MergedNumericTablePtr;

} // namespace data_management
} // namespace daal

#endif
