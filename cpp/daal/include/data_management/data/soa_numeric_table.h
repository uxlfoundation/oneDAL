/* file: soa_numeric_table.h */
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
//  Implementation of a heterogeneous table stored as a structure of arrays.
//--
*/

#ifndef __SOA_NUMERIC_TABLE_H__
#define __SOA_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/factory.h" // goes after the numeric_table.h to avoid circular dependency
#include "data_management/data/internal/conversion.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SOANUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a structure of arrays,
 *         where each (contiguous) array represents values corresponding to a specific feature.
 */
class DAAL_EXPORT SOANumericTable : public NumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    friend Creator<SOANumericTable>;

    /**
     *  Constructs an empty Numeric Table
     *  \param[in]  nColumns      Number of columns in the table
     *  \param[in]  nRows         Number of rows in the table
     *  \param[in]  featuresEqual Flag that makes all features in the NumericTableDictionary equal
     *  \param[out] stat          Status of the numeric table construction
     *  \return Empty numeric table
     */
    static SOANumericTable * create(size_t nColumns = 0, size_t nRows = 0, DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual,
                                    services::Status * stat = NULL);

    /**
     *  Constructs an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Shared pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                  Status of the numeric table construction
     *  \return     Numeric table with a predefined NumericTableDictionary
     */
    static SOANumericTable * create(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag = notAllocate,
                                    services::Status * stat = NULL);

    virtual ~SOANumericTable() { freeDataMemoryImpl(); }

    /**
     *  Sets a pointer to an array of values for a given feature
     *  \tparam T       Type of feature values
     *  \param[in]  ptr Pointer to the array of the T type that stores feature values
     *  \param[in]  idx Feature index
     */
    template <typename T>
    services::Status setArray(const services::SharedPtr<T> & ptr, size_t idx)
    {
        if (_partialMemStatus != notAllocated && _partialMemStatus != userAllocated)
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        if (idx < getNumberOfColumns() && idx < _arrays.size())
        {
            _ddict->setFeature<T>(idx);

            if (!_arrays[idx] && ptr)
            {
                _arraysInitialized++;
            }

            if (_arrays[idx] && !ptr)
            {
                _arraysInitialized--;
            }

            _arrays[idx] = services::reinterpretPointerCast<byte, T>(ptr);
        }
        else
        {
            return services::Status(services::ErrorIncorrectNumberOfFeatures);
        }

        _partialMemStatus = userAllocated;

        if (_arraysInitialized == getNumberOfColumns())
        {
            _memStatus = userAllocated;
        }
        DAAL_CHECK_STATUS_VAR(generatesOffsets())
        return services::Status();
    }

    /**
    *  Sets a pointer to an array of values for a given feature
    *  \tparam T       Type of feature values
    *  \param[in]  ptr Pointer to the array of the T type that stores feature values
    *  \param[in]  idx Feature index
    */
    template <typename T>
    services::Status setArray(T * ptr, size_t idx)
    {
        return setArray(services::SharedPtr<T>(ptr, services::EmptyDeleter()), idx);
    }

    /**
     *  Returns a pointer to an array of values for a given feature
     *  \param[in]  idx Feature index
     *  \return Pointer to the array of values
     */
    services::SharedPtr<byte> getArraySharedPtr(size_t idx)
    {
        if (idx < _ddict->getNumberOfFeatures())
        {
            return _arrays[idx];
        }
        else
        {
            this->_status.add(services::ErrorIncorrectNumberOfFeatures);
            return services::SharedPtr<byte>();
        }
    }

    /**
     *  Returns a pointer to an array of values for a given feature
     *  \param[in]  idx Feature index
     *  \return Pointer to the array of values
     */
    void * getArray(size_t idx) { return getArraySharedPtr(idx).get(); }

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

    DAAL_DEPRECATED_VIRTUAL services::Status setDictionary(NumericTableDictionary * ddict) DAAL_C11_OVERRIDE
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, NumericTable::setDictionary(ddict));

        size_t ncol = ddict->getNumberOfFeatures();

        if (!resizePointersArray(ncol))
        {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }
        return s;
    }

    /**
     *  Returns 'true' if all features have the same data type, else 'false'
     *  \return All features have the same data type or not
     */
    bool isHomogeneousFloatOrDouble() const;

protected:
    /**
     *  Constructor for an empty Numeric Table
     *  \param[in]  nColumns      Number of columns in the table
     *  \param[in]  nRows         Number of rows in the table
     *  \param[in]  featuresEqual Flag that makes all features in the NumericTableDictionary equal
     *  \DAAL_DEPRECATED_USE{ SOANumericTable::create }
     */
    SOANumericTable(size_t nColumns = 0, size_t nRows = 0, DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual);

    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED SOANumericTable(NumericTableDictionary * ddict, size_t nRows, AllocationFlag memoryAllocationFlag = notAllocate);

    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict                 Shared pointer to the predefined NumericTableDictionary
     *  \param[in]  nRows                 Number of rows in the table
     *  \param[in]  memoryAllocationFlag  Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED_USE{ SOANumericTable::create }
     */
    SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag = notAllocate);

    /**
     *  <a name="DAAL-CLASS-DATA_MANAGEMENT__WRAPPEDRAWPOINTER"></a>
     *  \brief   Class that provides functionality of deep copy.
     */
    class WrappedRawPointer
    {
    public:
        WrappedRawPointer() : _arrOffsets(NULL), _count(0) {};
        WrappedRawPointer(const WrappedRawPointer & wrapper)
        {
            allocate(wrapper._count);
            const size_t size = _count * sizeof(DAAL_INT64);
            daal::services::internal::daal_memcpy_s(_arrOffsets, size, wrapper._arrOffsets, size);
        }

        ~WrappedRawPointer() { deallocate(); }

        WrappedRawPointer & operator=(WrappedRawPointer const & wrapper)
        {
            if (this == &wrapper) return *this;

            if (_count < wrapper._count)
            {
                if (allocate(wrapper._count) != services::Status()) return *this;
            }
            else
            {
                _count = wrapper._count;
            }

            const size_t size = _count * sizeof(DAAL_INT64);
            daal::services::internal::daal_memcpy_s(_arrOffsets, size, wrapper._arrOffsets, size);

            return *this;
        }

        services::Status allocate(size_t count)
        {
            deallocate();
            _arrOffsets = (DAAL_INT64 *)daal::services::daal_malloc(count * sizeof(DAAL_INT64));
            DAAL_CHECK_MALLOC(_arrOffsets)
            _count = count;
            return services::Status();
        }

        void deallocate()
        {
            if (_arrOffsets)
            {
                daal::services::daal_free(_arrOffsets);
                _arrOffsets = NULL;
                _count      = 0;
            }
        }

        DAAL_INT64 const * get() const { return _arrOffsets; }
        DAAL_INT64 * get() { return _arrOffsets; }
        size_t count() const { return _count; }

    protected:
        DAAL_INT64 * _arrOffsets;
        size_t _count;
    };

    SOANumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status & st);

    SOANumericTable(NumericTableDictionaryPtr ddict, size_t nRows, AllocationFlag memoryAllocationFlag, services::Status & st);

    services::Collection<services::SharedPtr<byte> > _arrays;
    size_t _arraysInitialized;
    MemoryStatus _partialMemStatus;
    WrappedRawPointer _wrapOffsets;
    size_t _index;

    bool resizePointersArray(size_t nColumns);
    services::Status setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE;

    services::Status allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) DAAL_C11_OVERRIDE;

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE;

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        if (onDeserialize)
        {
            allocateDataMemoryImpl();
        }

        size_t ncol  = _ddict->getNumberOfFeatures();
        size_t nrows = getNumberOfRows();

        for (size_t i = 0; i < ncol; i++)
        {
            NumericTableFeature f = (*_ddict)[i];
            void * ptr            = getArraySharedPtr(i).get();

            arch->set((char *)ptr, nrows * f.typeSize);
        }

        return services::Status();
    }

    services::Status generatesOffsets()
    {
        if (isAllCompleted() && isHomogeneousFloatOrDouble())
        {
            DAAL_CHECK_STATUS_VAR(searchMinPointer());
        }

        return services::Status();
    }

private:
    bool isAllCompleted() const;

    services::Status searchMinPointer();

protected:
    template <typename T>
    DAAL_FORCEINLINE services::Status getTBlock(size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (!(block.getRWFlag() & (int)readOnly)) return services::Status();

        T * buffer    = block.getBlockPtr();
        bool computed = false;

        if (_wrapOffsets.get())
        {
            NumericTableFeature & f = (*_ddict)[0];
            if (daal::data_management::features::getIndexNumType<T>() == f.indexType)
            {
                T const * ptrMin = (T *)(_arrays[_index].get()) + idx;
                computed         = data_management::internal::getVector<T>()(nrows, ncols, buffer, ptrMin, _wrapOffsets.get());
            }
        }
        if (!computed)
        {
            constexpr size_t unrollFactor = 32; // size of the buffer for temporary storage of values of one feature after up-casting
            size_t di                     = unrollFactor;
            T lbuf[unrollFactor];

            for (size_t i = 0; i < nrows; i += di)
            {
                if (i + di > nrows)
                {
                    di = nrows - i;
                }

                for (size_t j = 0; j < ncols; ++j)
                {
                    NumericTableFeature & f = (*_ddict)[j];

                    char * ptr = (char *)_arrays[j].get() + (idx + i) * f.typeSize;

                    internal::getVectorUpCast(f.indexType, internal::getConversionDataType<T>())(di, ptr, lbuf);

                    for (size_t k = 0; k < di; ++k)
                    {
                        buffer[(i + k) * ncols + j] = lbuf[k];
                    }
                }
            }
        }

        return services::Status();
    }

    template <typename T>
    DAAL_FORCEINLINE services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();
            size_t nrows = block.getNumberOfRows();
            size_t idx   = block.getRowsOffset();

            constexpr size_t unrollFactor = 32; // size of the buffer for temporary storage of values of one feature before down-casting
            T lbuf[unrollFactor];
            size_t di = unrollFactor;

            T * blockPtr = block.getBlockPtr();

            for (size_t i = 0; i < nrows; i += di)
            {
                if (i + di > nrows)
                {
                    di = nrows - i;
                }

                for (size_t j = 0; j < ncols; j++)
                {
                    NumericTableFeature & f = (*_ddict)[j];

                    char * ptr = (char *)_arrays[j].get() + (idx + i) * f.typeSize;

                    for (size_t ii = 0; ii < di; ii++)
                    {
                        lbuf[ii] = blockPtr[(i + ii) * ncols + j];
                    }

                    internal::getVectorDownCast(f.indexType, internal::getConversionDataType<T>())(di, lbuf, ptr);
                }
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    DAAL_FORCEINLINE services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        size_t nobs = getNumberOfRows();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        const NumericTableFeature & f = (*_ddict)[feat_idx];
        const int indexType           = f.indexType;

        if (features::internal::getIndexNumType<T>() == f.indexType)
        {
            block.setPtr(&(_arrays[feat_idx]), _arrays[feat_idx].get() + idx * f.typeSize, 1, nrows);
        }
        else
        {
            if (data_management::features::DAAL_OTHER_T == indexType)
            {
                block.reset();
                return services::Status(services::ErrorDataTypeNotSupported);
            }

            byte * location = _arrays[feat_idx].get() + idx * f.typeSize;

            if (!(block.getRWFlag() & (int)readOnly)) return services::Status();

            internal::getVectorUpCast(indexType, internal::getConversionDataType<T>())(nrows, location, block.getBlockPtr());
        }
        return services::Status();
    }

    template <typename T>
    DAAL_FORCEINLINE services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature & f = (*_ddict)[feat_idx];
            const int indexType     = f.indexType;

            if (data_management::features::DAAL_OTHER_T == indexType)
            {
                block.reset();
                return services::Status(services::ErrorDataTypeNotSupported);
            }

            if (features::internal::getIndexNumType<T>() != indexType)
            {
                char * ptr = (char *)_arrays[feat_idx].get() + block.getRowsOffset() * f.typeSize;

                internal::getVectorDownCast(indexType, internal::getConversionDataType<T>())(block.getNumberOfRows(), block.getBlockPtr(), ptr);
            }
        }
        block.reset();
        return services::Status();
    }
};
typedef services::SharedPtr<SOANumericTable> SOANumericTablePtr;
/** @} */
} // namespace interface1
using interface1::SOANumericTable;
using interface1::SOANumericTablePtr;

} // namespace data_management
} // namespace daal
#endif
