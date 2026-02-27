/* file: row_merged_numeric_table.h */
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
//  Implementation of row merged numeric table.
//--
*/

#ifndef __ROW_MERGED_NUMERIC_TABLE_H__
#define __ROW_MERGED_NUMERIC_TABLE_H__

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__ROWMERGEDNUMERICTABLE"></a>
 *  \brief Class that provides methods to access a collection of numeric tables as if they are joined by rows
 */
class DAAL_EXPORT RowMergedNumericTable : public NumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    /**
     * Constructor for an empty merge Numeric Table
     * \param[out] stat  Status of the RowMergedNumericTable construction
     */
    static services::SharedPtr<RowMergedNumericTable> create(services::Status * stat = NULL);

    /**
     * Constructor for an empty merge Numeric Table
     * \param[in]  nestedTable  Pointer to the table
     * \param[out] stat         Status of the RowMergedNumericTable construction
     */
    static services::SharedPtr<RowMergedNumericTable> create(const NumericTablePtr & nestedTable, services::Status * stat = NULL);

    /**
     *  Adds the table to the bottom of the Row Merged Numeric Table
     *  \param[in] table Pointer to the table
     */
    services::Status addNumericTable(NumericTablePtr table)
    {
        if (table->getDataLayout() & csrArray) return services::Status(services::ErrorIncorrectTypeOfInputNumericTable);

        size_t ncols = getNumberOfColumns();
        size_t cols  = table->getNumberOfColumns();

        if (ncols != 0 && ncols != cols) return services::Status(services::ErrorIncorrectNumberOfFeatures);

        _tables->push_back(table);

        if (ncols == 0)
        {
            DictionaryIface::FeaturesEqual featuresEqual = table->getDictionarySharedPtr()->getFeaturesEqual();
            services::Status s;
            _ddict = NumericTableDictionary::create(ncols, featuresEqual, &s);
            if (!s) return s;
            s = setNumberOfColumnsImpl(cols);
            if (!s) return s;
            if (featuresEqual == DictionaryIface::equal)
            {
                NumericTableFeature & f = table->getDictionarySharedPtr()->operator[](0);
                _ddict->setFeature(f, 0);
            }
            else
            {
                for (size_t i = 0; i < cols; i++)
                {
                    NumericTableFeature & f = table->getDictionarySharedPtr()->operator[](i);
                    _ddict->setFeature(f, i);
                }
            }
        }

        size_t obs = table->getNumberOfRows();
        return setNumberOfRowsImpl(_obsnum + obs);
    }

    services::Status resize(size_t /*nrows*/) DAAL_C11_OVERRIDE
    {
        return services::Status(services::throwIfPossible(services::ErrorMethodNotSupported));
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
    void internal_inner_repack(size_t idx, size_t rows, size_t ncols, T * src, T * dst)
    {
        size_t i, j;

        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < ncols; j++)
            {
                dst[(idx + i) * ncols + j] = src[i * ncols + j];
            }
        }
    }

    template <typename T>
    void internal_outer_repack(size_t idx, size_t rows, size_t ncols, T * src, T * dst)
    {
        size_t i, j;

        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < ncols; j++)
            {
                dst[i * ncols + j] = src[(i + idx) * ncols + j];
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
            size_t rows = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lrows      = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd   = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;
                    s |= nt->getBlockOfRows(idxBegin - rows, idxEnd - idxBegin, readOnly, innerBlock);

                    internal_inner_repack<T>(idxBegin - idx, idxEnd - idxBegin, ncols, innerBlock.getBlockPtr(), block.getBlockPtr());

                    s |= nt->releaseBlockOfRows(innerBlock);
                }

                rows += lrows;
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
            size_t ncols = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx   = block.getRowsOffset();
            size_t rows  = 0;
            BlockDescriptor<T> innerBlock;
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lrows      = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd   = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;
                    s |= nt->getBlockOfRows(idxBegin - rows, idxEnd - idxBegin, writeOnly, innerBlock);

                    internal_outer_repack<T>(idxBegin - idx, idxEnd - idxBegin, ncols, block.getBlockPtr(), innerBlock.getBlockPtr());

                    s |= nt->releaseBlockOfRows(innerBlock);
                }

                rows += lrows;
            }
        }
        block.reset();
        return s;
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        services::Status s;
        const size_t nobs = getNumberOfRows();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (rwFlag & (int)readOnly)
        {
            T * buffer  = block.getBlockPtr();
            size_t rows = 0;
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lrows      = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd   = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;

                    BlockDescriptor<T> innerBlock;
                    s |= nt->getBlockOfColumnValues(feat_idx, idxBegin - rows, idxEnd - idxBegin, readOnly, innerBlock);
                    T * location = innerBlock.getBlockPtr();
                    for (size_t i = idxBegin; i < idxEnd; i++)
                    {
                        buffer[i] = location[i - idxBegin];
                    }
                    s |= nt->releaseBlockOfColumnValues(innerBlock);
                }

                rows += lrows;
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
            size_t rows     = 0;
            T * buffer      = block.getBlockPtr();
            for (size_t k = 0; k < _tables->size() && rows < idx + nrows; k++)
            {
                NumericTable * nt = (NumericTable *)(_tables->operator[](k).get());
                size_t lrows      = nt->getNumberOfRows();

                if (rows + lrows > idx)
                {
                    size_t idxBegin = (rows < idx) ? idx : rows;
                    size_t idxEnd   = (rows + lrows < idx + nrows) ? rows + lrows : idx + nrows;

                    BlockDescriptor<T> innerBlock;
                    s |= nt->getBlockOfColumnValues(feat_idx, idxBegin - rows, idxEnd - idxBegin, writeOnly, innerBlock);
                    T * location = innerBlock.getBlockPtr();
                    for (size_t i = idxBegin; i < idxEnd; i++)
                    {
                        location[i - idxBegin] = buffer[i];
                    }
                    s |= nt->releaseBlockOfColumnValues(innerBlock);
                }

                rows += lrows;
            }
        }
        block.reset();
        return s;
    }

    services::Status setNumberOfColumnsImpl(size_t ncols) DAAL_C11_OVERRIDE;
    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE;
    void freeDataMemoryImpl() DAAL_C11_OVERRIDE;

protected:
    DataCollectionPtr _tables;

    /**
     *  Constructor for an empty merge Numeric Table
     *  \DAAL_DEPRECATED_USE{ MergedNumericTable::create }
     */
    RowMergedNumericTable();

    /**
     *  Constructor for a Row Merged Numeric Table consisting of one table
     *  \param[in]  table  Pointer to the table
     *  \DAAL_DEPRECATED_USE{ MergedNumericTable::create }
     */
    RowMergedNumericTable(NumericTablePtr table);

    RowMergedNumericTable(services::Status & st);

    RowMergedNumericTable(const NumericTablePtr & table, services::Status & st);
};
typedef services::SharedPtr<RowMergedNumericTable> RowMergedNumericTablePtr;
/** @} */
} // namespace interface1
using interface1::RowMergedNumericTable;
using interface1::RowMergedNumericTablePtr;

} // namespace data_management
} // namespace daal

#endif
