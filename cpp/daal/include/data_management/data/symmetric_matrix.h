/* file: symmetric_matrix.h */
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
//  Declaration and implementation of a symmetric matrix.
//--
*/

#ifndef __SYMMETRIC_MATRIX_H__
#define __SYMMETRIC_MATRIX_H__

#include "services/daal_memory.h"
#include "services/daal_defines.h"

#include "data_management/data/data_serialize.h"
#include "data_management/data/numeric_table.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__PACKEDARRAYNUMERICTABLEIFACE"></a>
 *  \brief Abstract class that defines the interface of symmetric matrices stored as a one-dimensional array
 */
class PackedArrayNumericTableIface
{
public:
    virtual ~PackedArrayNumericTableIface() {}
    /**
     *  Gets the whole packed array of a requested data type
     *
     *  \param[in]  rwflag  Flag specifying read/write access to a block of feature vectors.
     *  \param[out] block   The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<double> & block) = 0;

    /**
     *  Gets the whole packed array of a requested data type
     *
     *  \param[in]  rwflag  Flag specifying read/write access to a block of feature vectors.
     *  \param[out] block   The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<float> & block) = 0;

    /**
     *  Gets the whole packed array of a requested data type
     *
     *  \param[in]  rwflag  Flag specifying read/write access to a block of feature vectors.
     *  \param[out] block   The block of feature values.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<int> & block) = 0;

    /**
     *  Releases a packed array
     *  \param[in] block   The block of feature values.
     */
    virtual services::Status releasePackedArray(BlockDescriptor<double> & block) = 0;

    /**
     *  Releases a packed array
     *  \param[in] block   The block of feature values.
     */
    virtual services::Status releasePackedArray(BlockDescriptor<float> & block) = 0;

    /**
     *  Releases a packed array
     *  \param[in] block   The block of feature values.
     */
    virtual services::Status releasePackedArray(BlockDescriptor<int> & block) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__PACKEDSYMMETRICMATRIX"></a>
 *  \brief Class that provides methods to access symmetric matrices stored as a one-dimensional array.
 *  \tparam DataType Defines the underlying data type that describes the Numeric Table
 *  \DAAL_DEPRECATED
 */
template <NumericTableIface::StorageLayout packedLayout, typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT PackedSymmetricMatrix : public NumericTable, public PackedArrayNumericTableIface
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    /**
     *  Typedef that stores the datatype used for template instantiation
     */
    typedef DataType baseDataType;

public:
    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedSymmetricMatrix(DataType * const ptr = 0, size_t nDim = 0) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()));

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedSymmetricMatrix(const services::SharedPtr<DataType> & ptr, size_t nDim) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(ptr);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    DAAL_DEPRECATED PackedSymmetricMatrix(size_t nDim /*= 0*/) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[out] stat        Status of the table construction
     *  \return Numeric table with user-allocated memory
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED static services::SharedPtr<PackedSymmetricMatrix> create(DataType * const ptr = 0, size_t nDim = 0,
                                                                             services::Status * stat = NULL)
    {
        return create(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nDim, stat);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[out] stat        Status of the table construction
     *  \return Numeric table with user-allocated memory
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED static services::SharedPtr<PackedSymmetricMatrix> create(const services::SharedPtr<DataType> & ptr, size_t nDim,
                                                                             services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(PackedSymmetricMatrix, DAAL_TEMPLATE_ARGUMENTS(packedLayout, DataType), ptr, nDim);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedSymmetricMatrix(DataType * const ptr, size_t nDim, const DataType & constValue) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()));

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        this->_status |= assign(constValue);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedSymmetricMatrix(const services::SharedPtr<DataType> & ptr, size_t nDim, const DataType & constValue)
        : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(ptr);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        this->_status |= assign(constValue);
    }

    /**
     *  Constructs a numeric table with user-allocated memory and fills the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     *  \param[out] stat        Status of the table construction
     *  \return     Numeric table with user-allocated memory initialized with a constant
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED static services::SharedPtr<PackedSymmetricMatrix> create(DataType * const ptr, size_t nDim, const DataType & constValue,
                                                                             services::Status * stat = NULL)
    {
        return create(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nDim, constValue, stat);
    }

    /**
     *  Constructs a numeric table with user-allocated memory and fills the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     *  \param[out] stat        Status of the table construction
     *  \return     Numeric table with user-allocated memory initialized with a constant
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED static services::SharedPtr<PackedSymmetricMatrix> create(const services::SharedPtr<DataType> & ptr, size_t nDim,
                                                                             const DataType & constValue, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(PackedSymmetricMatrix, DAAL_TEMPLATE_ARGUMENTS(packedLayout, DataType), ptr, nDim, constValue);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedSymmetricMatrix(size_t nDim, AllocationFlag memoryAllocationFlag) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) this->_status |= allocateDataMemoryImpl();
    }

    /**
     *  Constructs a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                    Status of the table construction
     *  \return     Numeric table
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED static services::SharedPtr<PackedSymmetricMatrix> create(size_t nDim, AllocationFlag memoryAllocationFlag,
                                                                             services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(PackedSymmetricMatrix, DAAL_TEMPLATE_ARGUMENTS(packedLayout, DataType), nDim, memoryAllocationFlag);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the packed symmetric matrix
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedSymmetricMatrix(size_t nDim, NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue)
        : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) this->_status |= allocateDataMemoryImpl();
        this->_status |= assign(constValue);
    }

    /**
     *  Constructs a Numeric Table with memory allocation controlled via a flag and fills the table with a constant
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the packed symmetric matrix
     *  \param[out] stat                    Status of the table construction
     *  \return     Numeric table initialized with a constant
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED static services::SharedPtr<PackedSymmetricMatrix> create(size_t nDim, NumericTable::AllocationFlag memoryAllocationFlag,
                                                                             const DataType & constValue, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(PackedSymmetricMatrix, DAAL_TEMPLATE_ARGUMENTS(packedLayout, DataType), nDim, memoryAllocationFlag,
                                             constValue);
    }

    /** \private */
    virtual ~PackedSymmetricMatrix() { freeDataMemoryImpl(); }

    /**
     *  Returns a pointer to a data set registered in the packed symmetric matrix
     *  \return Pointer to the data set
     */
    DataType * getArray() const { return (DataType *)_ptr.get(); }

    /**
     *  Returns a pointer to a data set registered in the packed symmetric matrix
     *  \return Pointer to the data set
     */
    services::SharedPtr<DataType> getArraySharedPtr() const { return services::reinterpretPointerCast<DataType, byte>(_ptr); }

    /**
     *  Sets a pointer to a packed array
     *  \param[in] ptr Pointer to the data set in the packed format
     */
    services::Status setArray(DataType * const ptr)
    {
        freeDataMemoryImpl();
        if (ptr == NULL) return services::Status(services::ErrorEmptyHomogenNumericTable);

        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        return services::Status();
    }

    /**
     *  Sets a pointer to a packed array
     *  \param[in] ptr Pointer to the data set in the packed format
     */
    services::Status setArray(const services::SharedPtr<DataType> & ptr)
    {
        freeDataMemoryImpl();
        if (ptr.get() == NULL) return services::Status(services::ErrorEmptyHomogenNumericTable);

        _ptr       = services::reinterpretPointerCast<byte, DataType>(ptr);
        _memStatus = userAllocated;
        return services::Status();
    }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  value  Constant to initialize entries of the packed symmetric matrix
     */
    template <typename T>
    services::Status assign(T value)
    {
        if (_memStatus == notAllocated) return services::Status(services::ErrorEmptyHomogenNumericTable);

        const size_t nDim = getNumberOfColumns();

        DataType * ptr         = (DataType *)_ptr.get();
        DataType valueDataType = (DataType)value;
        for (size_t i = 0; i < (nDim * (nDim + 1)) / 2; i++)
        {
            ptr[i] = valueDataType;
        }
        return services::Status();
    }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(float value) DAAL_C11_OVERRIDE { return assign<DataType>((DataType)value); }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(double value) DAAL_C11_OVERRIDE { return assign<DataType>((DataType)value); }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(int value) DAAL_C11_OVERRIDE { return assign<DataType>((DataType)value); }

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

    services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTPackedArray<double>(rwflag, block);
    }
    services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTPackedArray<float>(rwflag, block);
    }
    services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTPackedArray<int>(rwflag, block);
    }

    services::Status releasePackedArray(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTPackedArray<double>(block); }
    services::Status releasePackedArray(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTPackedArray<float>(block); }
    services::Status releasePackedArray(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTPackedArray<int>(block); }

protected:
    services::SharedPtr<byte> _ptr;

    PackedSymmetricMatrix(const services::SharedPtr<DataType> & ptr, size_t nDim, services::Status & st)
        : NumericTable(nDim, nDim, DictionaryIface::notEqual, st)
    {
        _layout = packedLayout;
        st |= setArray(ptr);

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);
    }

    PackedSymmetricMatrix(const services::SharedPtr<DataType> & ptr, size_t nDim, const DataType & constValue, services::Status & st)
        : NumericTable(nDim, nDim, DictionaryIface::notEqual, st)
    {
        _layout = packedLayout;
        st |= setArray(ptr);

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);

        st |= assign(constValue);
    }

    PackedSymmetricMatrix(size_t nDim, AllocationFlag memoryAllocationFlag, services::Status & st)
        : NumericTable(nDim, nDim, DictionaryIface::notEqual, st)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) st |= allocateDataMemoryImpl();
    }

    PackedSymmetricMatrix(size_t nDim, NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue, services::Status & st)
        : NumericTable(nDim, nDim, DictionaryIface::notEqual, st)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) st |= allocateDataMemoryImpl();
        st |= assign(constValue);
    }

    services::Status allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();

        const size_t nDim = getNumberOfColumns();
        const size_t size = (nDim * (nDim + 1)) / 2;

        if (size == 0)
            return services::Status(getNumberOfColumns() == 0 ? services::ErrorIncorrectNumberOfFeatures :
                                                                services::ErrorIncorrectNumberOfObservations);

        _ptr = services::SharedPtr<byte>((byte *)daal::services::daal_malloc(size * sizeof(DataType)), services::ServiceDeleter());

        if (_ptr.get() == NULL) return services::Status(services::ErrorMemoryAllocationFailed);

        _memStatus = internallyAllocated;
        return services::Status();
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _ptr.reset();
        _memStatus = notAllocated;
    }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        if (onDeserialize)
        {
            allocateDataMemoryImpl();
        }

        const size_t nDim = getNumberOfColumns();
        const size_t size = (nDim * (nDim + 1)) / 2;

        arch->set((DataType *)_ptr.get(), size);

        return services::Status();
    }

private:
    template <typename T1, typename T2>
    services::Status internal_repack(size_t p, size_t n, T1 * src, T2 * dst)
    {
        if (IsSameType<T1, T2>::value)
        {
            if (src != (T1 *)dst)
            {
                int result = daal::services::internal::daal_memcpy_s(dst, n * p * sizeof(T1), src, n * p * sizeof(T1));
                DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
            }
        }
        else
        {
            size_t i, j;

            for (i = 0; i < n; i++)
            {
                for (j = 0; j < p; j++)
                {
                    dst[i * p + j] = static_cast<T2>(src[i * p + j]);
                }
            }
        }
        return services::Status();
    }

    template <typename T1, typename T2>
    void internal_set_col_repack(size_t p, size_t n, T1 * src, T2 * dst)
    {
        size_t i;

        for (i = 0; i < n; i++)
        {
            dst[i * p] = static_cast<T2>(src[i]);
        }
    }

protected:
    baseDataType & getBaseValue(size_t dim, size_t rowIdx, size_t colIdx)
    {
        size_t offset;

        if (packedLayout == upperPackedSymmetricMatrix)
        {
            if (colIdx < rowIdx)
            {
                size_t tmp;
                tmp    = colIdx;
                colIdx = rowIdx;
                rowIdx = tmp;
            }

            offset = (2 * dim - rowIdx) * (rowIdx + 1) / 2 - (dim - colIdx);
        }
        else /* here lowerPackedSymmetricMatrix is supposed */
        {
            if (colIdx > rowIdx)
            {
                size_t tmp;
                tmp    = colIdx;
                colIdx = rowIdx;
                rowIdx = tmp;
            }

            offset = (2 + rowIdx) * (rowIdx + 1) / 2 - (rowIdx - colIdx) - 1;
        }
        return *((DataType *)_ptr.get() + offset);
    }

    template <typename T>
    T getValue(size_t dim, size_t rowIdx, size_t colIdx)
    {
        return static_cast<T>(getBaseValue(dim, rowIdx, colIdx));
    }

    template <typename T>
    services::Status setValue(size_t dim, size_t rowIdx, size_t colIdx, T value)
    {
        getBaseValue(dim, rowIdx, colIdx) = static_cast<baseDataType>(value);
        return services::Status();
    }

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        const size_t nDim = getNumberOfColumns();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nDim)
        {
            block.resizeBuffer(nDim, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nDim) ? nrows : nDim - idx;

        if (!block.resizeBuffer(nDim, nrows)) return services::Status(services::ErrorMemoryAllocationFailed);

        if ((rwFlag & (int)readOnly))
        {
            T * buffer = block.getBlockPtr();

            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                for (size_t iCol = 0; iCol < nDim; iCol++)
                {
                    buffer[iRow * nDim + iCol] = getValue<T>(nDim, iRow + idx, iCol);
                }
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        services::Status s;
        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t nDim  = getNumberOfColumns();
            const size_t nrows = block.getNumberOfRows();
            const size_t idx   = block.getRowsOffset();
            T * buffer         = block.getBlockPtr();

            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                for (size_t iCol = 0; iCol < nDim; iCol++)
                {
                    s |= setValue<T>(nDim, idx + iRow, iCol, buffer[iRow * nDim + iCol]);
                }
            }
        }
        block.reset();
        return s;
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        const size_t nDim = getNumberOfColumns();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nDim)
        {
            block.resizeBuffer(nDim, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nDim) ? nrows : nDim - idx;

        if (!block.resizeBuffer(1, nrows)) return services::Status();

        if ((block.getRWFlag() & (int)readOnly))
        {
            T * buffer = block.getBlockPtr();

            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                buffer[iRow] = getValue<T>(nDim, iRow + idx, feat_idx);
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        services::Status s;
        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t nDim     = getNumberOfColumns();
            const size_t nrows    = block.getNumberOfRows();
            const size_t idx      = block.getRowsOffset();
            const size_t feat_idx = block.getColumnsOffset();
            T * buffer            = block.getBlockPtr();

            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                s |= setValue<T>(nDim, iRow + idx, feat_idx, buffer[iRow]);
            }
        }
        block.reset();
        return s;
    }

    template <typename T>
    services::Status getTPackedArray(int rwFlag, BlockDescriptor<T> & block)
    {
        const size_t nDim = getNumberOfColumns();
        block.setDetails(0, 0, rwFlag);

        const size_t nSize = (nDim * (nDim + 1)) / 2;

        if (IsSameType<T, DataType>::value)
        {
            block.setPtr(&_ptr, _ptr.get(), 1, nSize);
            return services::Status();
        }

        if (!block.resizeBuffer(1, nSize)) return services::Status();

        if (!(rwFlag & (int)readOnly)) return services::Status();

        T * buffer     = block.getBlockPtr();
        DataType * ptr = (DataType *)_ptr.get();
        for (size_t i = 0; i < nSize; i++)
        {
            buffer[i] = static_cast<T>(*(ptr + i));
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTPackedArray(BlockDescriptor<T> & block)
    {
        if ((block.getRWFlag() & (int)writeOnly) && !IsSameType<T, DataType>::value)
        {
            const size_t nDim  = getNumberOfColumns();
            const size_t nSize = (nDim * (nDim + 1)) / 2;
            T * buffer         = block.getBlockPtr();
            DataType * ptr     = (DataType *)_ptr.get();

            for (size_t i = 0; i < nSize; i++)
            {
                *(ptr + i) = static_cast<baseDataType>(buffer[i]);
            }
        }
        block.reset();
        return services::Status();
    }

    virtual services::Status setNumberOfColumnsImpl(size_t nDim) DAAL_C11_OVERRIDE
    {
        if (_ddict->getNumberOfFeatures() != nDim)
        {
            _ddict->setNumberOfFeatures(nDim);

            NumericTableFeature df;
            df.setType<DataType>();
            _ddict->setAllFeatures(df);
        }

        _obsnum = nDim;
        return services::Status();
    }

    virtual services::Status setNumberOfRowsImpl(size_t nDim) DAAL_C11_OVERRIDE
    {
        setNumberOfColumnsImpl(nDim);
        _obsnum = nDim;
        return services::Status();
    }
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__PACKEDTRIANGULARMATRIX"></a>
 *  \brief Class that provides methods to access a packed triangular matrix stored as a one-dimensional array.
 *  \tparam DataType Defines the underlying data type that describes the packed triangular matrix
 *  \DAAL_DEPRECATED
 */
template <NumericTableIface::StorageLayout packedLayout, typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT PackedTriangularMatrix : public NumericTable, public PackedArrayNumericTableIface
{
public:
    DECLARE_SERIALIZABLE_TAG()

    /**
     *  Typedef that stores the data type used for template instantiation
     */
    typedef DataType baseDataType;

public:
    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedTriangularMatrix(DataType * const ptr = 0, size_t nDim = 0) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()));

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedTriangularMatrix(const services::SharedPtr<DataType> & ptr, size_t nDim) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(ptr);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    DAAL_DEPRECATED PackedTriangularMatrix(size_t nDim /*= 0*/) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    DAAL_DEPRECATED static services::SharedPtr<PackedTriangularMatrix> create(DataType * const ptr = 0, size_t nDim = 0,
                                                                              services::Status * stat = NULL)
    {
        return create(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nDim, stat);
    }

    DAAL_DEPRECATED static services::SharedPtr<PackedTriangularMatrix> create(const services::SharedPtr<DataType> & ptr, size_t nDim,
                                                                              services::Status * stat = NULL)
    {
        services::SharedPtr<PackedTriangularMatrix> ntPtr(new PackedTriangularMatrix(nDim));
        if (ntPtr.get())
        {
            services::Status s = ntPtr->setArray(ptr);
            if (!s) ntPtr = services::SharedPtr<PackedTriangularMatrix>();
            if (stat) *stat = s;
        }
        else
        {
            if (stat) *stat = services::Status(services::ErrorMemoryAllocationFailed);
        }
        return ntPtr;
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedTriangularMatrix(DataType * const ptr, size_t nDim, const DataType & constValue) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()));

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        this->_status |= assign(constValue);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  nDim        Matrix dimension
     *  \param[in]  constValue  Constant to initialize entries of the packed symmetric matrix
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedTriangularMatrix(const services::SharedPtr<DataType> & ptr, size_t nDim, const DataType & constValue)
        : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;
        this->_status |= setArray(ptr);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        this->_status |= assign(constValue);
    }

    DAAL_DEPRECATED static services::SharedPtr<PackedTriangularMatrix> create(DataType * const ptr, size_t nDim, const DataType & constValue,
                                                                              services::Status * stat = NULL)
    {
        services::SharedPtr<PackedTriangularMatrix> ntPtr(new PackedTriangularMatrix(nDim));
        if (ntPtr.get())
        {
            services::Status s = ntPtr->setArray(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()));
            s.add(ntPtr->assign(constValue));
            if (!s) ntPtr = services::SharedPtr<PackedTriangularMatrix>();
            if (stat) *stat = s;
        }
        else
        {
            if (stat) *stat = services::Status(services::ErrorMemoryAllocationFailed);
        }
        return ntPtr;
    }

    DAAL_DEPRECATED static services::SharedPtr<PackedTriangularMatrix> create(services::SharedPtr<DataType> & ptr, size_t nDim,
                                                                              const DataType & constValue, services::Status * stat = NULL)
    {
        services::SharedPtr<PackedTriangularMatrix> ntPtr(new PackedTriangularMatrix(nDim));
        if (ntPtr.get())
        {
            services::Status s = ntPtr->setArray(ptr);
            s.add(ntPtr->assign(constValue));
            if (!s) ntPtr = services::SharedPtr<PackedTriangularMatrix>();
            if (stat) *stat = s;
        }
        else
        {
            if (stat) *stat = services::Status(services::ErrorMemoryAllocationFailed);
        }
        return ntPtr;
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedTriangularMatrix(size_t nDim, AllocationFlag memoryAllocationFlag) : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) this->_status |= allocateDataMemoryImpl();
    }

    DAAL_DEPRECATED static services::SharedPtr<PackedTriangularMatrix> create(size_t nDim, AllocationFlag memoryAllocationFlag,
                                                                              services::Status * stat = NULL)
    {
        services::SharedPtr<PackedTriangularMatrix> ntPtr(new PackedTriangularMatrix(nDim));
        if (ntPtr.get())
        {
            services::Status s;
            if (memoryAllocationFlag == doAllocate)
            {
                s = ntPtr->allocateDataMemoryImpl();
                if (!s) ntPtr = services::SharedPtr<PackedTriangularMatrix>();
            }
            if (!s) ntPtr = services::SharedPtr<PackedTriangularMatrix>();
            if (stat) *stat = s;
        }
        else
        {
            if (stat) *stat = services::Status(services::ErrorMemoryAllocationFailed);
        }
        return ntPtr;
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  nDim                    Matrix dimension
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the packed symmetric matrix
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED PackedTriangularMatrix(size_t nDim, NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue)
        : NumericTable(nDim, nDim)
    {
        _layout = packedLayout;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) this->_status |= allocateDataMemoryImpl();
        this->_status |= assign(constValue);
    }

    DAAL_DEPRECATED static services::SharedPtr<PackedTriangularMatrix> create(size_t nDim, NumericTable::AllocationFlag memoryAllocationFlag,
                                                                              const DataType & constValue, services::Status * stat = NULL)
    {
        services::SharedPtr<PackedTriangularMatrix> ntPtr(new PackedTriangularMatrix(nDim));
        if (ntPtr.get())
        {
            services::Status s;
            if (memoryAllocationFlag == doAllocate)
            {
                s = ntPtr->allocateDataMemoryImpl();
                s.add(ntPtr->assign(constValue));
                if (!s) ntPtr = services::SharedPtr<PackedTriangularMatrix>();
            }
            if (!s) ntPtr = services::SharedPtr<PackedTriangularMatrix>();
            if (stat) *stat = s;
        }
        else
        {
            if (stat) *stat = services::Status(services::ErrorMemoryAllocationFailed);
        }
        return ntPtr;
    }

    /** \private */
    virtual ~PackedTriangularMatrix() { freeDataMemoryImpl(); }

    virtual services::Status setNumberOfColumns(size_t nDim) DAAL_C11_OVERRIDE
    {
        if (_ddict->getNumberOfFeatures() != nDim)
        {
            _ddict->setNumberOfFeatures(nDim);

            NumericTableFeature df;
            df.setType<DataType>();
            _ddict->setAllFeatures(df);
        }

        _obsnum = nDim;
        return services::Status();
    }

    virtual services::Status setNumberOfRows(size_t nDim) DAAL_C11_OVERRIDE { return setNumberOfColumns(nDim); }

    /**
     *  Returns a pointer to a data set registered in the packed symmetric matrix
     *  \return Pointer to the data set
     */
    DataType * getArray() const { return (DataType *)_ptr.get(); }

    /**
     *  Returns a pointer to a data set registered in the packed symmetric matrix
     *  \return Pointer to the data set
     */
    services::SharedPtr<DataType> getArraySharedPtr() const { return services::reinterpretPointerCast<DataType, byte>(_ptr); }

    /**
     *  Sets a pointer to an array that stores a packed triangular matrix
     *  \param[in] ptr Pointer to the array that stores the packed triangular matrix
     */
    services::Status setArray(DataType * const ptr)
    {
        freeDataMemoryImpl();
        if (ptr == NULL) return services::Status(services::ErrorEmptyHomogenNumericTable);

        _ptr       = services::SharedPtr<byte>((DataType *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        return services::Status();
    }

    /**
     *  Sets a pointer to a packed array
     *  \param[in] ptr Pointer to the data set in the packed format
     */
    services::Status setArray(const services::SharedPtr<DataType> & ptr)
    {
        freeDataMemoryImpl();
        if (ptr.get() == NULL) return services::Status(services::ErrorEmptyHomogenNumericTable);

        _ptr       = services::reinterpretPointerCast<byte, DataType>(ptr);
        _memStatus = userAllocated;
        return services::Status();
    }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  value  Constant to initialize entries of the packed symmetric matrix
     */
    template <typename T>
    services::Status assign(T value)
    {
        if (_memStatus == notAllocated) return services::Status(services::ErrorEmptyHomogenNumericTable);

        const size_t nDim = getNumberOfColumns();

        DataType * ptr         = (DataType *)_ptr.get();
        DataType valueDataType = (DataType)value;
        for (size_t i = 0; i < (nDim * (nDim + 1)) / 2; i++)
        {
            ptr[i] = valueDataType;
        }
        return services::Status();
    }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(float value) DAAL_C11_OVERRIDE { return assign<DataType>((DataType)value); }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(double value) DAAL_C11_OVERRIDE { return assign<DataType>((DataType)value); }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(int value) DAAL_C11_OVERRIDE { return assign<DataType>((DataType)value); }

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

    services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTPackedArray<double>(rwflag, block);
    }
    services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTPackedArray<float>(rwflag, block);
    }
    services::Status getPackedArray(ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTPackedArray<int>(rwflag, block);
    }

    services::Status releasePackedArray(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE { return releaseTPackedArray<double>(block); }
    services::Status releasePackedArray(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTPackedArray<float>(block); }
    services::Status releasePackedArray(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTPackedArray<int>(block); }

    /** \private */
    services::Status serializeImpl(InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    /** \private */
    services::Status deserializeImpl(const OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

protected:
    services::SharedPtr<byte> _ptr;

    services::Status allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) DAAL_C11_OVERRIDE
    {
        freeDataMemoryImpl();

        const size_t nDim = getNumberOfColumns();
        const size_t size = (nDim * (nDim + 1)) / 2;

        if (size == 0)
            return services::Status(getNumberOfColumns() == 0 ? services::ErrorIncorrectNumberOfFeatures :
                                                                services::ErrorIncorrectNumberOfObservations);

        _ptr = services::SharedPtr<byte>((byte *)daal::services::daal_malloc(size * sizeof(DataType)), services::ServiceDeleter());

        if (_ptr.get() == NULL) return services::Status(services::ErrorMemoryAllocationFailed);

        _memStatus = internallyAllocated;
        return services::Status();
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _ptr.reset();
        _memStatus = notAllocated;
    }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        if (onDeserialize)
        {
            allocateDataMemoryImpl();
        }

        const size_t nDim = getNumberOfColumns();
        const size_t size = (nDim * (nDim + 1)) / 2;

        arch->set((DataType *)_ptr.get(), size);

        return services::Status();
    }

private:
    template <typename T1, typename T2>
    services::Status internal_repack(size_t p, size_t n, T1 * src, T2 * dst)
    {
        if (IsSameType<T1, T2>::value)
        {
            if (src != (T1 *)dst)
            {
                int result = daal::services::internal::daal_memcpy_s(dst, n * p * sizeof(T1), src, n * p * sizeof(T1));
                DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
            }
        }
        else
        {
            size_t i, j;

            for (i = 0; i < n; i++)
            {
                for (j = 0; j < p; j++)
                {
                    dst[i * p + j] = static_cast<T2>(src[i * p + j]);
                }
            }
        }
        return services::Status();
    }

    template <typename T1, typename T2>
    void internal_set_col_repack(size_t p, size_t n, T1 * src, T2 * dst)
    {
        size_t i;

        for (i = 0; i < n; i++)
        {
            dst[i * p] = static_cast<T2>(src[i]);
        }
    }

protected:
    baseDataType & getBaseValue(size_t dim, size_t rowIdx, size_t colIdx, baseDataType & zero)
    {
        size_t offset;

        if (packedLayout == upperPackedTriangularMatrix)
        {
            if (colIdx < rowIdx)
            {
                return zero;
            }

            offset = (2 * dim - rowIdx) * (rowIdx + 1) / 2 - (dim - colIdx);
        }
        else /* here lowerPackedTriangularMatrix is supposed */
        {
            if (colIdx > rowIdx)
            {
                return zero;
            }

            offset = (2 + rowIdx) * (rowIdx + 1) / 2 - (rowIdx - colIdx) - 1;
        }
        return *((DataType *)_ptr.get() + offset);
    }

    template <typename T>
    T getValue(size_t dim, size_t rowIdx, size_t colIdx)
    {
        baseDataType zero = (baseDataType)0;
        return static_cast<T>(getBaseValue(dim, rowIdx, colIdx, zero));
    }

    template <typename T>
    services::Status setValue(size_t dim, size_t rowIdx, size_t colIdx, T value)
    {
        baseDataType zero                       = (baseDataType)0;
        getBaseValue(dim, rowIdx, colIdx, zero) = static_cast<baseDataType>(value);
        return services::Status();
    }

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        const size_t nDim = getNumberOfColumns();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nDim)
        {
            block.resizeBuffer(nDim, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nDim) ? nrows : nDim - idx;

        if (!block.resizeBuffer(nDim, nrows)) return services::Status(services::ErrorMemoryAllocationFailed);

        if ((rwFlag & (int)readOnly))
        {
            T * buffer = block.getBlockPtr();

            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                for (size_t iCol = 0; iCol < nDim; iCol++)
                {
                    buffer[iRow * nDim + iCol] = getValue<T>(nDim, iRow + idx, iCol);
                }
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        services::Status s;
        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t nDim  = getNumberOfColumns();
            const size_t nrows = block.getNumberOfRows();
            const size_t idx   = block.getRowsOffset();
            T * buffer         = block.getBlockPtr();

            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                for (size_t iCol = 0; iCol < nDim; iCol++)
                {
                    s |= setValue<T>(nDim, iRow + idx, iCol, buffer[iRow * nDim + iCol]);
                }
            }
        }
        block.reset();
        return s;
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        const size_t nDim = getNumberOfColumns();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nDim)
        {
            block.resizeBuffer(nDim, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nDim) ? nrows : nDim - idx;

        if (!block.resizeBuffer(1, nrows)) return services::Status();

        if ((block.getRWFlag() & (int)readOnly))
        {
            T * buffer = block.getBlockPtr();
            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                buffer[iRow] = getValue<T>(nDim, iRow + idx, feat_idx);
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        services::Status s;
        if (block.getRWFlag() & (int)writeOnly)
        {
            const size_t nDim     = getNumberOfColumns();
            const size_t nrows    = block.getNumberOfRows();
            const size_t idx      = block.getRowsOffset();
            const size_t feat_idx = block.getColumnsOffset();
            T * buffer            = block.getBlockPtr();

            for (size_t iRow = 0; iRow < nrows; iRow++)
            {
                s |= setValue<T>(nDim, iRow + idx, feat_idx, buffer[iRow]);
            }
        }
        block.reset();
        return s;
    }

    template <typename T>
    services::Status getTPackedArray(int rwFlag, BlockDescriptor<T> & block)
    {
        const size_t nDim = getNumberOfColumns();
        block.setDetails(0, 0, rwFlag);

        const size_t nSize = (nDim * (nDim + 1)) / 2;

        if (IsSameType<T, DataType>::value)
        {
            block.setPtr(&_ptr, _ptr.get(), 1, nSize);
            return services::Status();
        }

        if (!block.resizeBuffer(1, nSize)) return services::Status();

        if (!(rwFlag & (int)readOnly)) return services::Status();

        T * buffer     = block.getBlockPtr();
        DataType * ptr = (DataType *)_ptr.get();
        for (size_t i = 0; i < nSize; i++)
        {
            buffer[i] = static_cast<T>(*(ptr + i));
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTPackedArray(BlockDescriptor<T> & block)
    {
        if ((block.getRWFlag() & (int)writeOnly) && !IsSameType<T, DataType>::value)
        {
            const size_t nDim  = getNumberOfColumns();
            const size_t nSize = (nDim * (nDim + 1)) / 2;
            T * buffer         = block.getBlockPtr();
            DataType * ptr     = (DataType *)_ptr.get();

            for (size_t i = 0; i < nSize; i++)
            {
                *(ptr + i) = static_cast<baseDataType>(buffer[i]);
            }
        }
        block.reset();
        return services::Status();
    }

    virtual services::Status setNumberOfColumnsImpl(size_t nDim) DAAL_C11_OVERRIDE
    {
        if (_ddict->getNumberOfFeatures() != nDim)
        {
            _ddict->setNumberOfFeatures(nDim);

            NumericTableFeature df;
            df.setType<DataType>();
            _ddict->setAllFeatures(df);
        }

        _obsnum = nDim;
        return services::Status();
    }

    virtual services::Status setNumberOfRowsImpl(size_t nDim) DAAL_C11_OVERRIDE
    {
        setNumberOfColumnsImpl(nDim);
        _obsnum = nDim;
        return services::Status();
    }
};
/** @} */
} // namespace interface1
using interface1::PackedArrayNumericTableIface;
using interface1::PackedSymmetricMatrix;
using interface1::PackedTriangularMatrix;

} // namespace data_management
} // namespace daal
#endif
