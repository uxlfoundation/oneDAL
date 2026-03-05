/* file: arrow_numeric_table.h */
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
//  Implementation of a numeric table stored as a Apache Arrow table.
//--
*/

#ifndef __ARROW_NUMERIC_TABLE_H__
#define __ARROW_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/internal/conversion.h"
#include "data_management/data/internal/base_arrow_numeric_table.h"
#include <memory>
#include <arrow/table.h>
#include <arrow/util/config.h>

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__ARROWIMMUTABLENUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a Apache Arrow table.
 */
class DAAL_EXPORT ArrowImmutableNumericTable : public BaseArrowImmutableNumericTable
{
public:
    DECLARE_SERIALIZABLE_IMPL();

    /**
     *  Constructs an empty Numeric Table
     *  \param[in]  table         Apache Arrow table
     *  \param[out] stat          Status of the numeric table construction
     *  \return Empty numeric table
     */
    static DAAL_FORCEINLINE services::SharedPtr<ArrowImmutableNumericTable> create(const std::shared_ptr<arrow::Table> & table,
                                                                                   services::Status * stat = NULL)
    {
        if (!table)
        {
            if (stat)
            {
                stat->add(services::ErrorNullPtr);
            }
            return services::SharedPtr<ArrowImmutableNumericTable>();
        }

        DAAL_DEFAULT_CREATE_IMPL_EX(ArrowImmutableNumericTable, table);
    }

    /**
     *  Constructs an empty Numeric Table
     *  \param[in]  table         Apache Arrow table
     *  \param[out] stat          Status of the numeric table construction
     *  \return Empty numeric table
     */
    static DAAL_FORCEINLINE services::SharedPtr<ArrowImmutableNumericTable> create(const std::shared_ptr<const arrow::Table> & table,
                                                                                   services::Status * stat = NULL)
    {
        if (!table)
        {
            if (stat)
            {
                stat->add(services::ErrorNullPtr);
            }
            return services::SharedPtr<ArrowImmutableNumericTable>();
        }

        DAAL_DEFAULT_CREATE_IMPL_EX(ArrowImmutableNumericTable, table);
    }

    services::Status getBlockOfRows(size_t vectorIdx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vectorIdx, vector_num, rwflag, block);
    }

    services::Status getBlockOfRows(size_t vectorIdx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vectorIdx, vector_num, rwflag, block);
    }

    services::Status getBlockOfRows(size_t vectorIdx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vectorIdx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTBlock<double>(block); }

    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block); }

    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t featureIdx, size_t vectorIdx, size_t valueNum, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(featureIdx, vectorIdx, valueNum, rwflag, block);
    }

    services::Status getBlockOfColumnValues(size_t featureIdx, size_t vectorIdx, size_t valueNum, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(featureIdx, vectorIdx, valueNum, rwflag, block);
    }

    services::Status getBlockOfColumnValues(size_t featureIdx, size_t vectorIdx, size_t valueNum, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(featureIdx, vectorIdx, valueNum, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTFeature<int>(block); }

protected:
    services::Status setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE
    {
        if (ncol == getNumberOfColumns()) return services::Status();
        return services::Status(services::ErrorMethodNotSupported);
    }

    services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        return services::Status(services::ErrorMethodNotSupported);
    }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        if (onDeserialize)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }

        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        const size_t ncol = _ddict->getNumberOfFeatures();

        for (size_t i = 0; i < ncol; ++i)
        {
            const NumericTableFeature & f = (*_ddict)[i];

            const std::shared_ptr<const arrow::ChunkedArray> columnChunkedArrayPtr = getColumnChunkedArrayPtr(i);
            DAAL_ASSERT(columnChunkedArrayPtr);
            const arrow::ChunkedArray & columnChunkedArray = *columnChunkedArrayPtr;
            const int chunkCount                           = columnChunkedArray.num_chunks();
            DAAL_ASSERT(chunkCount > 0);

            for (int chunk = 0; chunk < chunkCount; ++chunk)
            {
                const std::shared_ptr<const arrow::Array> arrayPtr = columnChunkedArray.chunk(chunk);
                DAAL_ASSERT(arrayPtr);
                const int64_t chunkLength = arrayPtr->length();
                DAAL_ASSERT(chunkLength > 0);
                arch->set(getPtr(arrayPtr, f), chunkLength * f.typeSize);
            }
        }

        return services::Status();
    }

private:
    DAAL_FORCEINLINE ArrowImmutableNumericTable(const std::shared_ptr<const arrow::Table> & table, services::Status & st)
        : BaseArrowImmutableNumericTable(table->num_columns(), table->num_rows(), st), _table(table)
    {
        _layout    = arrow;
        _memStatus = userAllocated;
        if (st) st |= updateFeatures(*table);
    }

    std::shared_ptr<const arrow::Table> _table;

    DAAL_FORCEINLINE services::Status updateFeatures(const arrow::Table & table)
    {
        services::Status s;
        if (_ddict.get() == NULL)
        {
            _ddict = NumericTableDictionary::create(&s);
        }
        if (!s) return s;

        const std::shared_ptr<const arrow::Schema> schemaPtr = table.schema();
        DAAL_ASSERT(schemaPtr);
        const int ncols = schemaPtr->num_fields();
        for (int col = 0; col < ncols; ++col)
        {
            const arrow::Type::type type = schemaPtr->field(col)->type()->id();
            switch (type)
            {
            case arrow::Type::UINT8: s |= setFeature<unsigned char>(col); break;
            case arrow::Type::INT8: s |= setFeature<char>(col); break;
            case arrow::Type::UINT16: s |= setFeature<unsigned short>(col); break;
            case arrow::Type::INT16: s |= setFeature<short>(col); break;
            case arrow::Type::UINT32: s |= setFeature<unsigned int>(col); break;
            case arrow::Type::DATE32:
            case arrow::Type::TIME32:
            case arrow::Type::INT32: s |= setFeature<int>(col); break;
            case arrow::Type::UINT64: s |= setFeature<DAAL_UINT64>(col); break;
            case arrow::Type::DATE64:
            case arrow::Type::TIMESTAMP:
            case arrow::Type::TIME64:
            case arrow::Type::INT64: s |= setFeature<DAAL_INT64>(col); break;
            case arrow::Type::FLOAT: s |= setFeature<float>(col); break;
            case arrow::Type::DOUBLE: s |= setFeature<double>(col); break;
            default: s.add(services::ErrorDataTypeNotSupported); return s;
            }
        }
        return s;
    }

    template <typename T>
    services::Status setFeature(size_t idx, features::FeatureType featureType = features::DAAL_CONTINUOUS, size_t categoryNumber = 0)
    {
        DAAL_ASSERT(_ddict);
        services::Status s = _ddict->setFeature<T>(idx);
        if (!s) return s;
        (*_ddict)[idx].featureType    = featureType;
        (*_ddict)[idx].categoryNumber = categoryNumber;
        return s;
    }

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }

        const size_t ncols = getNumberOfColumns();
        const size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(ncols, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (!block.resizeBuffer(ncols, nrows))
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        T lbuf[32];
        size_t di        = 32;
        T * const buffer = block.getBlockPtr();

        for (size_t i = 0; i < nrows; i += di)
        {
            if (i + di > nrows)
            {
                di = nrows - i;
            }

            for (size_t j = 0; j < ncols; ++j)
            {
                const NumericTableFeature & f = (*_ddict)[j];

                const std::shared_ptr<const arrow::ChunkedArray> columnChunkedArrayPtr = getColumnChunkedArrayPtr(j);
                DAAL_ASSERT(columnChunkedArrayPtr);
                const std::shared_ptr<const arrow::ChunkedArray> sliceChunkedArrayPtr = columnChunkedArrayPtr->Slice(idx + i, di);
                DAAL_ASSERT(sliceChunkedArrayPtr);
                const arrow::ChunkedArray & sliceChunkedArray = *sliceChunkedArrayPtr;
                const int chunkCount                          = sliceChunkedArray.num_chunks();
                DAAL_ASSERT(chunkCount > 0);
                if (chunkCount == 1)
                {
                    const char * const ptr = getPtr(sliceChunkedArray.chunk(0), f);
                    DAAL_ASSERT(ptr);
                    internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())(di, ptr, lbuf);
                }
                else
                {
                    size_t offset = 0;
                    for (int chunk = 0; chunk < chunkCount; ++chunk)
                    {
                        const std::shared_ptr<const arrow::Array> arrayPtr = sliceChunkedArray.chunk(chunk);
                        DAAL_ASSERT(arrayPtr);
                        const int64_t chunkLength = arrayPtr->length();
                        DAAL_ASSERT(chunkLength > 0);
                        const char * const ptr = getPtr(arrayPtr, f);
                        DAAL_ASSERT(ptr);
                        internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())(chunkLength, ptr, &(lbuf[offset]));
                        offset += chunkLength;
                    }
                    DAAL_ASSERT(offset == di);
                }

                for (size_t ii = 0; ii < di; ++ii)
                {
                    buffer[(i + ii) * ncols + j] = lbuf[ii];
                }
            }
        }

        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }

        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t featIdx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }

        const size_t nobs = getNumberOfRows();
        block.setDetails(featIdx, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(1, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        const NumericTableFeature & f = (*_ddict)[featIdx];

        const std::shared_ptr<const arrow::ChunkedArray> columnChunkedArrayPtr = getColumnChunkedArrayPtr(featIdx);
        DAAL_ASSERT(columnChunkedArrayPtr);
        const std::shared_ptr<const arrow::ChunkedArray> sliceChunkedArrayPtr = columnChunkedArrayPtr->Slice(idx, nrows);
        DAAL_ASSERT(sliceChunkedArrayPtr);
        const arrow::ChunkedArray & sliceChunkedArray = *sliceChunkedArrayPtr;
        const int chunkCount                          = sliceChunkedArray.num_chunks();
        DAAL_ASSERT(chunkCount > 0);

        if (features::internal::getIndexNumType<T>() == f.indexType && chunkCount == 1)
        {
            const T * const ptr = getPtr<T>(sliceChunkedArray.chunk(0), f);
            DAAL_ASSERT(ptr);
            block.setPtr(const_cast<T * const>(ptr), 1, nrows);
        }
        else
        {
            if (!block.resizeBuffer(1, nrows))
            {
                return services::Status(services::ErrorMemoryAllocationFailed);
            }

            if (!(block.getRWFlag() & (int)readOnly)) return services::Status();

            if (chunkCount == 1)
            {
                const char * const ptr = getPtr(sliceChunkedArray.chunk(0), f);
                DAAL_ASSERT(ptr);
                internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())(nrows, ptr, block.getBlockPtr());
            }
            else
            {
                size_t offset     = 0;
                T * const destPtr = block.getBlockPtr();
                for (int chunk = 0; chunk < chunkCount; ++chunk)
                {
                    const std::shared_ptr<const arrow::Array> arrayPtr = sliceChunkedArray.chunk(chunk);
                    DAAL_ASSERT(arrayPtr);
                    const int64_t chunkLength = arrayPtr->length();
                    DAAL_ASSERT(chunkLength > 0);
                    const char * const ptr = getPtr(arrayPtr, f);
                    DAAL_ASSERT(ptr);
                    internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())(chunkLength, ptr, destPtr + offset);
                    offset += chunkLength;
                }
                DAAL_ASSERT(offset == di);
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            return services::Status(services::ErrorMethodNotSupported);
        }

        block.reset();
        return services::Status();
    }

    template <typename T = char>
    const T * getPtr(const arrow::Array & array, const NumericTableFeature & f, int bufferIndex = 1) const
    {
        const std::shared_ptr<const arrow::ArrayData> arrayDataPtr = array.data();
        DAAL_ASSERT(arrayDataPtr);
        const arrow::ArrayData & arrayData = *arrayDataPtr;
        return reinterpret_cast<const T *>(arrayData.template GetValues<char>(bufferIndex, arrayData.offset * f.typeSize));
    }

    template <typename T = char>
    const T * getPtr(const std::shared_ptr<const arrow::Array> & array, const NumericTableFeature & f, int bufferIndex = 1) const
    {
        return getPtr<T>(*array, f, bufferIndex);
    }

    const std::shared_ptr<const arrow::ChunkedArray> getColumnChunkedArrayPtr(size_t idx)
    {
#if ARROW_VERSION >= 15000
        return _table->column(idx);
#else
        const std::shared_ptr<const arrow::Column> columnPtr = _table->column(idx);
        DAAL_ASSERT(columnPtr);
        return columnPtr->data();
#endif
    }
};
typedef services::SharedPtr<ArrowImmutableNumericTable> ArrowImmutableNumericTablePtr;
/** @} */
} // namespace interface1
using interface1::ArrowImmutableNumericTable;
using interface1::ArrowImmutableNumericTablePtr;

} // namespace data_management
} // namespace daal
#endif
