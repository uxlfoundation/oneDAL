/* file: aos_numeric_table.h */
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

#ifndef __AOS_NUMERIC_TABLE_H__
#define __AOS_NUMERIC_TABLE_H__

#include "data_management/data/data_serialize.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/internal/conversion.h"
#include "services/daal_defines.h"

namespace daal
{
namespace data_management
{
// Extended variant of the standard offsetof() macro  (not limited to only POD types)
/* Not sure if it's standard-compliant; most likely, it only works in certain environments.
   The constant 0x1000 (not NULL) is necessary to appease GCC. */
#define DAAL_STRUCT_MEMBER_OFFSET(class_name, member_name) ((ptrdiff_t) & (reinterpret_cast<class_name *>(0x1000)->member_name) - 0x1000)

namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__AOSNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a contiguous array
 *         of heterogeneous feature vectors, while each feature vector is represented
 *         by a data structure.
 *         Therefore, the data is represented as an array of structures.
 */
class DAAL_EXPORT AOSNumericTable : public NumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    /**
     *  Constructor for an empty Numeric Table with a predefined size of the structure that represents a feature vector
     *  \param[in]  structSize  Size of the structure that represents the feature vector
     *  \param[in]  ncol        Number of columns in the table
     *  \param[in]  nrow        Number of rows in the table
     *  \DAAL_DEPRECATED_USE{ AOSNumericTable::create }
     */
    AOSNumericTable(size_t structSize = 0, size_t ncol = 0, size_t nrow = 0);

    /**
     *  Constructs an empty Numeric Table with a predefined size of the structure that represents a feature vector
     *  \param[in]  structSize  Size of the structure that represents the feature vector
     *  \param[in]  ncol        Number of columns in the table
     *  \param[in]  nrow        Number of rows in the table
     *  \param[out] stat        Status of the table construction
     *  \return Empty numeric table with a predefined size of the structure that represents a feature vector
     */
    static services::SharedPtr<AOSNumericTable> create(size_t structSize = 0, size_t ncol = 0, size_t nrow = 0, services::Status * stat = NULL);

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr     Pointer to a data set in the AOS format
     *  \param[in]  ncol    Number of columns in the table
     *  \param[in]  nrow    Number of rows in the table
     *  \DAAL_DEPRECATED_USE{ AOSNumericTable::create }
     */
    template <typename StructDataType>
    AOSNumericTable(const services::SharedPtr<StructDataType> & ptr, size_t ncol, size_t nrow = 0) : NumericTable(ncol, nrow)
    {
        _ptr        = services::reinterpretPointerCast<byte, StructDataType>(ptr);
        _layout     = aos;
        _structSize = sizeof(StructDataType);

        initOffsets();
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  ptr     Pointer to a data set in the AOS format
     *  \param[in]  ncol    Number of columns in the table
     *  \param[in]  nrow    Number of rows in the table
     *  \param[out] stat    Status of the table construction
     *  \return Numeric table with user-allocated memory
     */
    template <typename StructDataType>
    static services::SharedPtr<AOSNumericTable> create(const services::SharedPtr<StructDataType> & ptr, size_t ncol, size_t nrow = 0,
                                                       services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_IMPL_EX(AOSNumericTable, ptr, ncol, nrow);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr     Pointer to a data set in the AOS format
     *  \param[in]  ncol    Number of columns in the table
     *  \param[in]  nrow    Number of rows in the table
     *  \DAAL_DEPRECATED_USE{ AOSNumericTable::create }
     */
    template <typename StructDataType>
    AOSNumericTable(StructDataType * ptr, size_t ncol, size_t nrow = 0) : NumericTable(ncol, nrow)
    {
        _ptr        = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _layout     = aos;
        _structSize = sizeof(StructDataType);

        initOffsets();
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  ptr     Pointer to a data set in the AOS format
     *  \param[in]  ncol    Number of columns in the table
     *  \param[in]  nrow    Number of rows in the table
     *  \param[out] stat    Status of the table construction
     *  \return Numeric table with user-allocated memory
     */
    template <typename StructDataType>
    static services::SharedPtr<AOSNumericTable> create(StructDataType * ptr, size_t ncol, size_t nrow = 0, services::Status * /*stat*/ = NULL)
    {
        return create(services::SharedPtr<StructDataType>(ptr, services::EmptyDeleter()), ncol, nrow);
    }

    /** \private */
    virtual ~AOSNumericTable()
    {
        if (_offsets)
        {
            daal::services::daal_free(_offsets);
            _offsets = NULL;
        }
        freeDataMemoryImpl();
    }

    /**
     *  Sets a pointer to an array of structures in a Numeric Table
     *  \param[in]  ptr Pointer to a data set in the AOS format
     *  \param[in]  obsnum Number of rows in the table
     */
    services::Status setArray(void * const ptr, size_t obsnum = 0)
    {
        _ptr       = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _memStatus = userAllocated;
        return setNumberOfRowsImpl(obsnum);
    }

    /**
     *  Sets a pointer to an array of structures in a Numeric Table
     *  \param[in]  ptr Pointer to a data set in the AOS format
     *  \param[in]  obsnum Number of rows in the table
     */
    services::Status setArray(const services::SharedPtr<byte> & ptr, size_t obsnum = 0)
    {
        _ptr       = ptr;
        _memStatus = userAllocated;
        return setNumberOfRowsImpl(obsnum);
    }

    /**
     *  Returns a pointer to an array of structures in a Numeric Table
     *  \return Pointer to a data set in the AOS format
     */
    void * getArray() { return (void *)(_ptr.get()); }

    /**
     *  Returns a pointer to an array of structures in a Numeric Table
     *  \return Pointer to a data set in the AOS format
     */
    const void * getArray() const { return (void *)(_ptr.get()); }

    /**
     *  Returns a pointer to an array of structures in a Numeric Table
     *  \return Pointer to a data set in the AOS format
     */
    services::SharedPtr<byte> getArraySharedPtr() { return _ptr; }

    /**
     *  Sets a feature in an AOS Numeric Table
     *  \tparam     T              Type of feature values
     *  \param[in]  idx            Feature index
     *  \param[in]  offset         Feature offset in the structure representing the feature vector
     *  \param[in]  featureType    Feature type
     *  \param[in]  categoryNumber Number of categories for categorical features
     */
    template <typename T>
    services::Status setFeature(size_t idx, size_t offset, features::FeatureType featureType = features::DAAL_CONTINUOUS, size_t categoryNumber = 0)
    {
        if (offset >= _structSize || idx >= getNumberOfColumns())
        {
            return services::throwIfPossible(services::Status(services::ErrorIncorrectDataRange));
        }

        services::Status s;
        if (_ddict.get() == NULL)
        {
            _ddict = NumericTableDictionary::create(&s);
        }
        if (!s) return s;

        s = _ddict->setFeature<T>(idx);
        if (!s) return s;
        (*_ddict)[idx].featureType    = featureType;
        (*_ddict)[idx].categoryNumber = categoryNumber;

        _offsets[idx] = offset;
        return s;
    }

    /**
     *  Sets an offset in an AOS Numeric Table
     *  \param[in] idx      Feature index
     *  \param[in] offset   Feature offset in the structure representing the feature vector
     */
    void setOffset(size_t idx, size_t offset)
    {
        if (offset >= _structSize || idx >= getNumberOfColumns())
        {
            _status.add(services::throwIfPossible(services::Status(services::ErrorIncorrectDataRange)));
        }
        else
        {
            _offsets[idx] = offset;
        }
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
    services::SharedPtr<byte> _ptr;
    size_t _structSize;
    size_t * _offsets;

    AOSNumericTable(size_t structSize, size_t ncol, size_t nrow, services::Status & st);

    template <typename StructDataType>
    AOSNumericTable(const services::SharedPtr<StructDataType> & ptr, size_t ncol, size_t nrow, services::Status & st)
        : NumericTable(ncol, nrow, DictionaryIface::notEqual, st)
    {
        _ptr        = services::reinterpretPointerCast<byte, StructDataType>(ptr);
        _layout     = aos;
        _structSize = sizeof(StructDataType);

        st |= initOffsets();
    }

    services::Status allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) DAAL_C11_OVERRIDE
    {
        if (checkOffsets())
        {
            services::Status s = createOffsetsFromDictionary();
            if (!s) return s;
        }

        freeDataMemoryImpl();

        const size_t size = _structSize * getNumberOfRows();

        if (size == 0)
            return services::Status(getNumberOfRows() == 0 ? services::ErrorIncorrectNumberOfObservations : services::ErrorIncorrectNumberOfFeatures);

        _ptr = services::SharedPtr<byte>((byte *)daal::services::daal_malloc(size), services::ServiceDeleter());
        if (!_ptr) return services::Status(services::ErrorMemoryAllocationFailed);

        _memStatus = internallyAllocated;
        return services::Status();
    }

    bool checkOffsets() const
    {
        if (!_offsets) return true;

        const size_t ncols = getNumberOfColumns();

        size_t sizeOfRowInDict = 0;
        for (size_t i = 0; i < ncols; ++i)
        {
            if (!(*_ddict)[i].typeSize)
            {
                return false;
            }
            sizeOfRowInDict += (*_ddict)[i].typeSize;
        }
        if (sizeOfRowInDict > _structSize)
        {
            return true;
        }

        for (size_t i = 1; i < ncols; ++i)
        {
            if (_offsets[i - 1] >= _offsets[i])
            {
                return true;
            }
        }
        return false;
    }

    services::Status createOffsetsFromDictionary()
    {
        const size_t ncols = getNumberOfColumns();

        if (_offsets) daal::services::daal_free(_offsets);

        _offsets = (size_t *)daal::services::daal_malloc(sizeof(size_t) * (ncols));
        if (!_offsets) return services::Status(services::ErrorMemoryAllocationFailed);

        size_t offset = 0;
        for (size_t i = 0; i < ncols; ++i)
        {
            _offsets[i] = offset;
            offset += (*_ddict)[i].typeSize;
        }
        _structSize = offset;

        return services::Status();
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE
    {
        _ptr       = services::SharedPtr<byte>();
        _memStatus = notAllocated;
    }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        NumericTable::serialImpl<Archive, onDeserialize>(arch);
        arch->set(_structSize);

        if (onDeserialize)
        {
            initOffsets();
        }
        arch->set((char *)_offsets, getNumberOfColumns() * sizeof(size_t));

        if (onDeserialize)
        {
            allocateDataMemoryImpl();
        }

        size_t size = getNumberOfRows();

        arch->set((char *)_ptr.get(), size * _structSize);

        return services::Status();
    }

private:
    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (!(rwFlag & (int)readOnly)) return services::Status();

        char * ptr = (char *)(_ptr.get()) + _structSize * idx;

        for (size_t j = 0; j < ncols; j++)
        {
            NumericTableFeature & f = (*_ddict)[j];

            char * location = ptr + _offsets[j];

            T * blockPtr = block.getBlockPtr();

            internal::getVectorStrideUpCast(f.indexType, internal::getConversionDataType<T>())(nrows, location, _structSize, blockPtr + j,
                                                                                               sizeof(T) * ncols);
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols = getNumberOfColumns();

            char * ptr = (char *)(_ptr.get()) + _structSize * block.getRowsOffset();

            T * blockPtr = block.getBlockPtr();

            for (size_t j = 0; j < ncols; j++)
            {
                NumericTableFeature & f = (*_ddict)[j];

                char * location = ptr + _offsets[j];

                internal::getVectorStrideDownCast(f.indexType, internal::getConversionDataType<T>())(block.getNumberOfRows(), blockPtr + j,
                                                                                                     sizeof(T) * ncols, location, _structSize);
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        size_t nobs = getNumberOfRows();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if ((block.getRWFlag() & (int)readOnly))
        {
            NumericTableFeature & f = (*_ddict)[feat_idx];
            char * ptr              = (char *)(_ptr.get()) + _structSize * idx + _offsets[feat_idx];
            internal::getVectorStrideUpCast(f.indexType, internal::getConversionDataType<T>())(nrows, ptr, _structSize, block.getBlockPtr(),
                                                                                               sizeof(T));
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t feat_idx = block.getColumnsOffset();

            NumericTableFeature & f = (*_ddict)[feat_idx];

            char * ptr = (char *)(_ptr.get()) + _structSize * block.getRowsOffset() + _offsets[feat_idx];

            internal::getVectorStrideDownCast(f.indexType, internal::getConversionDataType<T>())(block.getNumberOfRows(), block.getBlockPtr(),
                                                                                                 sizeof(T), ptr, _structSize);
        }
        block.reset();
        return services::Status();
    }

    services::Status initOffsets()
    {
        const size_t ncols = getNumberOfColumns();
        if (ncols > 0)
        {
            _offsets = (size_t *)daal::services::daal_malloc(sizeof(size_t) * (ncols));
            if (!_offsets) return services::Status(services::ErrorMemoryAllocationFailed);
            for (size_t i = 0; i < ncols; ++i) _offsets[i] = 0;
        }
        else
        {
            _offsets = 0;
        }
        return services::Status();
    }
};
typedef services::SharedPtr<AOSNumericTable> AOSNumericTablePtr;
/** @} */
} // namespace interface1
using interface1::AOSNumericTable;
using interface1::AOSNumericTablePtr;

} // namespace data_management
} // namespace daal
#endif
