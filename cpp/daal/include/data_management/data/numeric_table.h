/* file: numeric_table.h */
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
//  Declaration and implementation of the base class for numeric tables.
//--
*/

#ifndef __NUMERIC_TABLE_H__
#define __NUMERIC_TABLE_H__

#include "services/base.h"
#include "services/internal/buffer.h"
#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "services/error_handling.h"
#include "algorithms/algorithm_types.h"
#include "data_management/data/data_collection.h"
#include "data_management/data/data_dictionary.h"

#include "data_management/data/numeric_types.h"

namespace daal
{
/** \brief Contains classes that implement data management functionality, including NumericTables, DataSources, and Compression */
namespace data_management
{
namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
class NumericTable;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__BLOCKDESCRIPTOR"></a>
 *  \brief %Base class that manages buffer memory for read/write operations required by numeric tables.
 */
template <typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT BlockDescriptor
{
public:
    /** \private */
    DAAL_FORCEINLINE BlockDescriptor()
        : _ptr(), _nrows(0), _ncols(0), _colsOffset(0), _rowsOffset(0), _rwFlag(0), _buffer(), _capacity(0), _pPtr(0), _rawPtr(0)
    {}

    /** \private */
    ~BlockDescriptor() { freeBuffer(); }

    /**
     *  Gets a pointer to the buffer
     *  \return Pointer to the block
     */
    inline DataType * getBlockPtr() const
    {
        if (_rawPtr)
        {
            return (DataType *)_rawPtr;
        }
        else if (_xBuffer)
        {
            return getCachedHostSharedPtr().get();
        }
        else
        {
            return _ptr.get();
        }
    }

    /**
     *  Gets a pointer to the buffer
     *  \return Pointer to the block
     */
    inline services::SharedPtr<DataType> getBlockSharedPtr() const
    {
        if (_rawPtr)
        {
            return services::SharedPtr<DataType>(services::reinterpretPointerCast<DataType, byte>(*_pPtr), (DataType *)_rawPtr);
        }
        else if (_xBuffer)
        {
            services::Status status;
            services::SharedPtr<DataType> ptr = _xBuffer.toHost((data_management::ReadWriteMode)_rwFlag, status);
            services::throwIfPossible(status);
            return ptr;
        }
        else
        {
            return _ptr;
        }
    }

#if (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0
    /**
     *  Gets a Buffer object to the data block
     *  \return Buffer to the block
     *  \DAAL_DEPRECATED
     */
    inline services::internal::Buffer<DataType> getBuffer() const
    {
        if (_rawPtr)
        {
            const size_t size = _ncols * _nrows;
            DAAL_ASSERT((size / _ncols) == _nrows);

            services::Status status;
            services::internal::Buffer<DataType> buffer((DataType *)_rawPtr, size, status);
            services::throwIfPossible(status);
            return buffer;
        }
        else if (_xBuffer)
        {
            return _xBuffer;
        }
        else
        {
            const size_t size = _ncols * _nrows;
            DAAL_ASSERT((size / _ncols) == _nrows);
            DAAL_ASSERT(_ptr.get() != nullptr);

            services::Status status;
            services::internal::Buffer<DataType> buffer(_ptr, size, status);
            services::throwIfPossible(status);
            return buffer;
        }
    }
#endif // (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0

    /**
     *  Returns the number of columns in the block
     *  \return Number of columns
     */
    inline size_t getNumberOfColumns() const
    {
        return _ncols;
    }

    /**
     *  Returns the number of rows in the block
     *  \return Number of rows
     */
    inline size_t getNumberOfRows() const
    {
        return _nrows;
    }

    /**
     * Resets internal values and pointers to zero values
     */
    inline void reset()
    {
        _colsOffset = 0;
        _rowsOffset = 0;
        _rwFlag     = 0;
        _pPtr       = NULL;
        _rawPtr     = NULL;
        _hostSharedPtr.reset();
    }

public:
    /**
     *  Sets data pointer to use for in-place calculation
     *  \param[in] ptr      Pointer to the buffer
     *  \param[in] nColumns Number of columns
     *  \param[in] nRows    Number of rows
     */
    inline void setPtr(DataType * ptr, size_t nColumns, size_t nRows)
    {
        _xBuffer.reset();
        _hostSharedPtr.reset();
        _ptr   = services::SharedPtr<DataType>(ptr, services::EmptyDeleter());
        _ncols = nColumns;
        _nrows = nRows;
    }

    /**
     *  \param[in] pPtr Pointer to the shared pointer that handles the memory
     *  \param[in] rawPtr Pointer to the shifted memory
     *  \param[in] nColumns Number of columns
     *  \param[in] nRows Number of rows
     */
    inline void setPtr(services::SharedPtr<byte> * pPtr, byte * rawPtr, size_t nColumns, size_t nRows)
    {
        _xBuffer.reset();
        _hostSharedPtr.reset();
        _pPtr   = pPtr;
        _rawPtr = rawPtr;
        _ncols  = nColumns;
        _nrows  = nRows;
    }

    /**
     *  Sets data pointer to use for in-place calculation
     *  \param[in] ptr      Shared pointer to the buffer
     *  \param[in] nColumns Number of columns
     *  \param[in] nRows    Number of rows
     */
    inline void setSharedPtr(const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows)
    {
        _xBuffer.reset();
        _hostSharedPtr.reset();
        _ptr   = ptr;
        _ncols = nColumns;
        _nrows = nRows;
    }

#if (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0
    /**
     *  Sets data buffer to the table
     *  \param[in] buffer Buffer object that contains the memory
     *  \param[in] nColumns Number of columns
     *  \param[in] nRows Number of rows
     *  \DAAL_DEPRECATED
     */
    inline void setBuffer(const daal::services::internal::Buffer<DataType> & buffer, size_t nColumns, size_t nRows)
    {
        _xBuffer = buffer;
        _hostSharedPtr.reset();
        _pPtr   = NULL;
        _rawPtr = NULL;
        _ncols  = nColumns;
        _nrows  = nRows;
    }

    /**
     *  Allocates memory of (\p nColumns * \p nRows + \p auxMemorySize) size
     *  \param[in] nColumns      Number of columns
     *  \param[in] nRows         Number of rows
     *  \param[in] auxMemorySize Memory size
     *
     *  \return true if memory of (\p nColumns * \p nRows + \p auxMemorySize) size is allocated successfully
     *  \DAAL_DEPRECATED
     */
    inline bool resizeBuffer(size_t nColumns, size_t nRows, size_t auxMemorySize = 0)
    {
        // TOOD: Resize _xBuffer
        _xBuffer.reset();
        _hostSharedPtr.reset();
        _ncols = nColumns;
        _nrows = nRows;

        const size_t elementsCount = nColumns * nRows;
        DAAL_ASSERT((elementsCount / nRows) == nColumns);

        const size_t bytesCount = elementsCount * sizeof(DataType);
        DAAL_ASSERT((bytesCount / sizeof(DataType)) == elementsCount);

        const size_t newSize = bytesCount + auxMemorySize;
        DAAL_ASSERT((newSize - bytesCount) == auxMemorySize);

        if (newSize > _capacity)
        {
            freeBuffer();
            _buffer = services::SharedPtr<DataType>((DataType *)daal::services::daal_malloc(newSize), services::ServiceDeleter());
            if (_buffer != 0)
            {
                _capacity = newSize;
            }
            else
            {
                return false;
            }
        }

        _ptr = _buffer;
        if (!auxMemorySize)
        {
            if (_aux_ptr)
            {
                _aux_ptr = services::SharedPtr<DataType>();
            }
        }
        else
        {
            _aux_ptr = services::SharedPtr<DataType>(_buffer, _buffer.get() + nColumns * nRows);
        }

        return true;
    }
#endif // (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0

    /**
     *  Sets parameters of the block
     *  \param[in]  columnIdx   Index of the first column in the block
     *  \param[in]  rowIdx      Index of the first row in the block
     *  \param[in]  rwFlag      Flag specifying read/write access to the block
     */
    inline void setDetails(size_t columnIdx, size_t rowIdx, int rwFlag)
    {
        _colsOffset = columnIdx;
        _rowsOffset = rowIdx;

        if (_rwFlag != rwFlag)
        {
            _rwFlag = rwFlag;
            _hostSharedPtr.reset(); // need to reallocate cached pointer when rwFlag is changed
        }
    }

    /**
     *  Gets the number of columns in the numeric table preceding the first element in the block
     *  \return columns offset
     */
    inline size_t getColumnsOffset() const
    {
        return _colsOffset;
    }

    /**
     *  Gets the number of rows in the numeric table preceding the first element in the block
     *  \return rows offset
     */
    inline size_t getRowsOffset() const
    {
        return _rowsOffset;
    }

    /**
     *  Gets the flag specifying read/write access to the block
     *  \return flag
     */
    inline size_t getRWFlag() const
    {
        return _rwFlag;
    }

#if (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0
    /**
     *  Gets a pointer to the additional memory buffer
     *  \return pointer
     *  \DAAL_DEPRECATED
     */
    void * getAdditionalBufferPtr() const
    {
        return _aux_ptr.get();
    }

    /**
     *  \DAAL_DEPRECATED
     */
    inline services::SharedPtr<DataType> getAdditionalBufferSharedPtr() const
    {
        return _aux_ptr;
    }
#endif // (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0

protected:
    /**
     *  Frees the buffer
     */
    void freeBuffer()
    {
        if (_buffer)
        {
            _buffer = services::SharedPtr<DataType>();
        }
        _capacity = 0;
    }

    /**
     *  Gets cached shared pointer to the block of memory from the Buffer object
     * \return shared pointer
     */

    inline services::SharedPtr<DataType> getCachedHostSharedPtr() const
    {
        if (!_hostSharedPtr)
        {
            services::Status status;
            _hostSharedPtr = _xBuffer.toHost((data_management::ReadWriteMode)_rwFlag, status);
            services::throwIfPossible(status);
        }
        return _hostSharedPtr;
    }

private:
    services::SharedPtr<DataType> _ptr;
    size_t _nrows;
    size_t _ncols;

    size_t _colsOffset;
    size_t _rowsOffset;
    int _rwFlag;

#if (INTEL_DAAL_VERSION < 202600000)        /// 2026.0.0
    services::SharedPtr<DataType> _aux_ptr; /*<! \DAAL_DEPRECATED */

    services::SharedPtr<DataType> _buffer; /*<! Pointer to the buffer. \DAAL_DEPRECATED */
    size_t _capacity;                      /*<! Buffer size in bytes. \DAAL_DEPRECATED */

    services::SharedPtr<byte> * _pPtr; /*<! \DAAL_DEPRECATED */
    byte * _rawPtr;                    /*<! \DAAL_DEPRECATED */

    daal::services::internal::Buffer<DataType> _xBuffer;  /*<! \DAAL_DEPRECATED */
    mutable services::SharedPtr<DataType> _hostSharedPtr; /*<! owns pointer returned from getBlockPtr() method \DAAL_DEPRECATED */
#endif                                                    // (INTEL_DAAL_VERSION < 202600000) /// 2026.0.0
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__NUMERICTABLEIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for representation of data in the numeric format.
 *  This class declares the most general methods for data access.
 */
class NumericTableIface
{
public:
    virtual ~NumericTableIface() {}
    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__MEMORYSTATUS"></a>
     * \brief Enumeration to specify the status of memory related to the Numeric Table
     */
    enum MemoryStatus
    {
        notAllocated,       /*!< No memory allocated */
        userAllocated,      /*!< Memory allocated on user side */
        internallyAllocated /*!< Memory allocated and managed by NumericTable */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__ALLOCATIONFLAG"></a>
     * \brief Enumeration to specify whether the Numeric Table must allocate memory
     */
    enum AllocationFlag
    {
        doNotAllocate = 0, /*!< Memory will not be allocated by NumericTable */
        notAllocate =
            0, /*!< Memory will not be allocated by NumericTable \DAAL_DEPRECATED_USE{ \ref daal::data_management::interface1::NumericTableIface::doNotAllocate "doNotAllocate" }*/
        doAllocate = 1 /*!< Memory will be allocated by NumericTable when needed */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__BASICSTATISTICSID"></a>
     * \brief Enumeration to specify estimates of basic statistics stored
     */
    enum BasicStatisticsId
    {
        minimum    = 0, /*!< Minimum estimate */
        maximum    = 1, /*!< Maximum estimate */
        sum        = 2, /*!< Sum estimate */
        sumSquares = 3  /*!< Sum squares estimate */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__FEATUREBASICSTATISTICS"></a>
     * \brief Enumeration to specify feature-specific estimates of basic statistics stored
     */
    enum FeatureBasicStatistics
    {
        counters /*!< Counters estimate */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__NORMALIZATIONTYPE"></a>
     * \brief Enumeration to specify types of normalization
     */
    enum NormalizationType
    {
        nonNormalized           = 0, /*!< Default: non-normalized */
        standardScoreNormalized = 1, /*!< Standard score normalization (mean=0, variance=1) */
        minMaxNormalized        = 2  /*!< Min-max normalization */
    };

    /**
     * <a name="DAAL-ENUM-DATA_MANAGEMENT__STORAGELAYOUT"></a>
     * \brief Storage layouts that may need to be supported
     */
    enum StorageLayout
    {
        soa                         = 1, // 1
        aos                         = 2, // 2
        csrArray                    = 1 << 4,
        upperPackedSymmetricMatrix  = 1 << 8,
        lowerPackedSymmetricMatrix  = 2 << 8,
        upperPackedTriangularMatrix = 1 << 7,
        lowerPackedTriangularMatrix = 4 << 8,
        arrow                       = 8 << 8,

        layout_unknown = 0x80000000 // the last bit set
    };

    /**
     *  Sets a data dictionary in the Numeric Table
     *  \param[in] ddict Pointer to the data dictionary
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status setDictionary(NumericTableDictionary * /*ddict*/) { return services::Status(); }

    /**
     *  Returns a pointer to a data dictionary
     *  \return Pointer to the data dictionary
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual NumericTableDictionary * getDictionary() const = 0;

    /**
     *  Returns a shared pointer to a data dictionary
     *  \return Shared pointer to the data dictionary
     */
    virtual NumericTableDictionaryPtr getDictionarySharedPtr() const = 0;

    /**
     *  Resets a data dictionary for the Numeric Table
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status resetDictionary() { return services::Status(); }

    /**
     *  Returns the type of a given feature
     *  \param[in] feature_idx Feature index
     *  \return Feature type
     */
    virtual features::FeatureType getFeatureType(size_t feature_idx) const = 0;

    /**
     *  Returns the number of categories for a given feature
     *  \param[in] feature_idx Feature index
     *  \return Number of categories
     */
    virtual size_t getNumberOfCategories(size_t feature_idx) const = 0;

    /**
     *  Returns a data layout used in the Numeric Table
     *  \return Data layout
     */
    virtual StorageLayout getDataLayout() const = 0;

    /**
     *  Sets the number of rows in the Numeric Table and allocates memory for a data set
     */
    virtual services::Status resize(size_t nrows) = 0;

    /**
     *  Sets the number of columns in the Numeric Table
     *
     *  \param[in] ncol Number of columns
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status setNumberOfColumns(size_t ncol) = 0;

    /**
     *  Sets the number of rows in the Numeric Table
     *
     *  \param[in] nrow Number of rows
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status setNumberOfRows(size_t nrow) = 0;

    /**
     *  Allocates memory for a data set
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual services::Status allocateDataMemory(daal::MemType type = daal::dram) = 0;

    /**
     *  Deallocates the memory allocated for a data set
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual void freeDataMemory() = 0;

    /**
     *  Allocates Numeric Tables for basic statistics
     */
    virtual services::Status allocateBasicStatistics() = 0;

    /**
     * Checks the correctness of this numeric table
     * \param[in] description           Additional information about error
     * \param[in] checkDataAllocation   Flag that specifies whether to check the data allocation status
     * \return                  Check status: True if the table satisfies the requirements, false otherwise.
     */
    virtual services::Status check(const char * description, bool checkDataAllocation = true) const = 0;
};
} // namespace interface1
using interface1::BlockDescriptor;
using interface1::NumericTableIface;

const int packed_mask = (int)NumericTableIface::csrArray | (int)NumericTableIface::upperPackedSymmetricMatrix
                        | (int)NumericTableIface::lowerPackedSymmetricMatrix | (int)NumericTableIface::upperPackedTriangularMatrix
                        | (int)NumericTableIface::lowerPackedTriangularMatrix;

namespace interface1
{
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DENSENUMERICTABLEIFACE"></a>
 *  \brief Abstract interface class for a data management component responsible for accessing data in the numeric format.
 *  This class declares specific methods to access data in a dense homogeneous form.
 */
class DenseNumericTableIface
{
public:
    virtual ~DenseNumericTableIface() {}
    /**
     *  Gets a block of rows from a table.
     *
     *  \param[in] vector_idx Index of the first row to include into the block.
     *  \param[in] vector_num Number of rows in the block.
     *  \param[in] rwflag     Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block     The block of feature vectors.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) = 0;

    /**
     *  Gets a block of rows from a table.
     *
     *  \param[in] vector_idx Index of the first row to include into the block.
     *  \param[in] vector_num Number of rows in the block.
     *  \param[in] rwflag     Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block     The block of feature vectors.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) = 0;

    /**
     *  Gets a block of rows from a table.
     *
     *  \param[in] vector_idx Index of the first row to include into the block.
     *  \param[in] vector_num Number of rows in the block.
     *  \param[in] rwflag     Flag specifying read/write access to the block of feature vectors.
     *  \param[out] block     The block of feature vectors.
     *
     *  \return Actual number of feature vectors returned by the method.
     */
    virtual services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) = 0;

    /**
     *  Releases a block of rows.
     *  \param[in] block      The block of rows.
     */
    virtual services::Status releaseBlockOfRows(BlockDescriptor<double> & block) = 0;

    /**
     *  Releases a block of rows.
     *  \param[in] block      The block of rows.
     */
    virtual services::Status releaseBlockOfRows(BlockDescriptor<float> & block) = 0;

    /**
     *  Releases a block of rows.
     *  \param[in] block      The block of rows.
     */
    virtual services::Status releaseBlockOfRows(BlockDescriptor<int> & block) = 0;

    /**
     *  Gets a block of values for a given feature.
     *
     *  \param[in] feature_idx Feature index.
     *  \param[in] vector_idx  Index of the first feature vector to include into the block.
     *  \param[in] value_num   Number of feature values in the block.
     *  \param[in] rwflag      Flag specifying read/write access to the block of feature values.
     *  \param[out] block      The block of feature values.
     *
     *  \return Actual number of feature values returned by the method.
     */
    virtual services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                                    BlockDescriptor<double> & block) = 0;

    /**
     *  Gets a block of values for a given feature.
     *
     *  \param[in] feature_idx Feature index.
     *  \param[in] vector_idx  Index of the first feature vector to include into the block.
     *  \param[in] value_num   Number of feature values in the block.
     *  \param[in] rwflag      Flag specifying read/write access to the block of feature values.
     *  \param[out] block      The block of feature values.
     *
     *  \return Actual number of feature values returned by the method.
     */
    virtual services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                                    BlockDescriptor<float> & block) = 0;

    /**
     *  Gets a block of values for a given feature.
     *
     *  \param[in] feature_idx Feature index.
     *  \param[in] vector_idx  Index of the first feature vector to include into the block.
     *  \param[in] value_num   Number of feature values in the block.
     *  \param[in] rwflag      Flag specifying read/write access to the block of feature values.
     *  \param[out] block      The block of feature values.
     *
     *  \return Actual number of feature values returned by the method.
     */
    virtual services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                                    BlockDescriptor<int> & block) = 0;

    /**
     *  Releases a block of values for a given feature.
     *  \param[in] block       The block of feature values.
     */
    virtual services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) = 0;

    /**
     *  Releases a block of values for a given feature.
     *  \param[in] block       The block of feature values.
     */
    virtual services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) = 0;

    /**
     *  Releases a block of values for a given feature.
     *  \param[in] block       The block of feature values.
     */
    virtual services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__NUMERICTABLE"></a>
 *  \brief Class for a data management component responsible for representation of data in the numeric format.
 *  This class implements the most general methods for data access.
 */
class DAAL_EXPORT NumericTable : public SerializationIface, public NumericTableIface, public DenseNumericTableIface
{
public:
    DAAL_CAST_OPERATOR(NumericTable)

    /**
     *  Constructor for a Numeric Table with predefined dictionary
     *  \param[in]  ddict          Pointer to the data dictionary
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED NumericTable(NumericTableDictionary * ddict)
    {
        _obsnum            = 0;
        _ddict             = NumericTableDictionaryPtr(ddict, services::EmptyDeleter());
        _layout            = layout_unknown;
        _memStatus         = notAllocated;
        _normalizationFlag = NumericTable::nonNormalized;
    }

    /**
     *  Constructor for a Numeric Table with predefined dictionary
     *  \param[in]  ddict          Pointer to the data dictionary
     */
    NumericTable(NumericTableDictionaryPtr ddict)
    {
        _obsnum            = 0;
        _ddict             = ddict;
        _layout            = layout_unknown;
        _memStatus         = notAllocated;
        _normalizationFlag = NumericTable::nonNormalized;
    }

    /**
     *  Constructor for a Numeric Table
     *  \param[in]  featnum        Number of columns in the table
     *  \param[in]  obsnum         Number of rows in the table
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     */
    NumericTable(size_t featnum, size_t obsnum, DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual)
    {
        _obsnum            = obsnum;
        _ddict             = NumericTableDictionaryPtr(new NumericTableDictionary(featnum, featuresEqual));
        _layout            = layout_unknown;
        _memStatus         = notAllocated;
        _normalizationFlag = NumericTable::nonNormalized;
    }

    /** \private */
    virtual ~NumericTable() {}

    DAAL_DEPRECATED_VIRTUAL virtual services::Status setDictionary(NumericTableDictionary * ddict) DAAL_C11_OVERRIDE
    {
        _ddict = NumericTableDictionaryPtr(ddict, services::EmptyDeleter());
        return services::Status();
    }

    DAAL_DEPRECATED_VIRTUAL virtual NumericTableDictionary * getDictionary() const DAAL_C11_OVERRIDE { return _ddict.get(); }

    virtual NumericTableDictionaryPtr getDictionarySharedPtr() const DAAL_C11_OVERRIDE { return _ddict; }

    DAAL_DEPRECATED_VIRTUAL virtual services::Status resetDictionary() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status resize(size_t nrows) DAAL_C11_OVERRIDE
    {
        size_t obsnum      = _obsnum;
        services::Status s = setNumberOfRowsImpl(nrows);
        if ((_memStatus != userAllocated && obsnum < nrows) || _memStatus == notAllocated)
        {
            s |= allocateDataMemoryImpl();
        }
        return s;
    }

    /**
     *  Returns the number of columns in the Numeric Table
     *  \return Number of columns
     */
    size_t getNumberOfColumns() const { return _ddict->getNumberOfFeatures(); }

    /**
     *  Returns the number of rows in the Numeric Table
     *  \return Number of rows
     */
    size_t getNumberOfRows() const { return _obsnum; }

    DAAL_DEPRECATED_VIRTUAL services::Status setNumberOfColumns(size_t ncol) DAAL_C11_OVERRIDE { return setNumberOfColumnsImpl(ncol); }

    DAAL_DEPRECATED_VIRTUAL services::Status setNumberOfRows(size_t nrow) DAAL_C11_OVERRIDE { return setNumberOfRowsImpl(nrow); }

    DAAL_DEPRECATED_VIRTUAL services::Status allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        return allocateDataMemoryImpl(type);
    }

    DAAL_DEPRECATED_VIRTUAL void freeDataMemory() DAAL_C11_OVERRIDE { freeDataMemoryImpl(); }

    StorageLayout getDataLayout() const DAAL_C11_OVERRIDE { return _layout; }

    features::FeatureType getFeatureType(size_t feature_idx) const DAAL_C11_OVERRIDE
    {
        if (_ddict.get() != NULL && _ddict->getNumberOfFeatures() > feature_idx)
        {
            const NumericTableFeature & f = (*_ddict)[feature_idx];
            return f.featureType;
        }
        else
        {
            /* If no dictionary was set, all features are considered numeric */
            return features::DAAL_CONTINUOUS;
        }
    }

    size_t getNumberOfCategories(size_t feature_idx) const DAAL_C11_OVERRIDE
    {
        if (_ddict.get() != NULL && _ddict->getNumberOfFeatures() > feature_idx && getFeatureType(feature_idx) != features::DAAL_CONTINUOUS)
        {
            const NumericTableFeature & f = (*_ddict)[feature_idx];
            return f.categoryNumber;
        }
        else
        {
            /* If no dictionary was set, all features are considered numeric */
            return (size_t)-1;
        }
    }

    /**
     *  Gets the status of the memory used by a data set connected with a Numeric Table
     */
    virtual MemoryStatus getDataMemoryStatus() const { return _memStatus; }

    /**
     *  Checks if dataset stored in the numeric table is normalized, according to the given normalization flag
     *  \param[in] flag Normalization flag to check
     *  \return Check result
     */
    bool isNormalized(NormalizationType flag) const { return (_normalizationFlag == flag); }

    /**
     *  Sets the normalization flag for dataset stored in the numeric table
     *  \param[in] flag Normalization flag
     *  \return Previous value of the normalization flag
     */
    NormalizationType setNormalizationFlag(NormalizationType flag)
    {
        NormalizationType oldValue = _normalizationFlag;
        _normalizationFlag         = flag;
        return oldValue;
    }

    /**
     *  Returns errors during the computation
     *  \return Errors during the computation
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::SharedPtr<services::KernelErrorCollection> getErrors() { return _status.getCollection()->getErrors(); }

    /**
     *  Allocates Numeric Tables for basic statistics
     */
    virtual services::Status allocateBasicStatistics() DAAL_C11_OVERRIDE;

    /**
     * \copydoc NumericTableIface::check
     */
    virtual services::Status check(const char * description, bool checkDataAllocation = true) const DAAL_C11_OVERRIDE
    {
        if (getDataMemoryStatus() == notAllocated && checkDataAllocation)
        {
            return services::Status(services::Error::create(services::ErrorNullNumericTable, services::ArgumentName, description));
        }

        if (getNumberOfColumns() == 0)
        {
            return services::Status(services::Error::create(services::ErrorIncorrectNumberOfColumns, services::ArgumentName, description));
        }

        if (getNumberOfRows() == 0 && getDataMemoryStatus() != notAllocated)
        {
            return services::Status(services::Error::create(services::ErrorIncorrectNumberOfRows, services::ArgumentName, description));
        }

        return services::Status();
    }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  value  Constant to initialize entries of the numeric table
     */
    virtual services::Status assign(float value) { return assignImpl<float>(value); }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  value  Constant to initialize entries of the numeric table
     */
    virtual services::Status assign(double value) { return assignImpl<double>(value); }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  value  Constant to initialize entries of the numeric table
     */
    virtual services::Status assign(int value) { return assignImpl<int>(value); }

    /**
     *  Returns value by given column and row from the numeric table
     *  \param[in]      column        Column
     *  \param[in]      row           Row
     *  \return Value from numeric table
     */
    template <typename DataType>
    DataType getValue(size_t column, size_t row) const
    {
        services::Status status;
        return getValueImpl<DataType>(column, row, status);
    }

    /**
     *  Returns value by given column and row from the numeric table
     *  \param[in]      column        Column
     *  \param[in]      row           Row
     *  \param[in,out]  status        Status of the operation
     *  \return Value from numeric table
     */
    template <typename DataType>
    DataType getValue(size_t column, size_t row, services::Status & status) const
    {
        return getValueImpl<DataType>(column, row, status);
    }

public:
    /**
     *  <a name="DAAL-CLASS-DATA_MANAGEMENT__BASICSTATISTICSDATACOLLECTION"></a>
     *  \brief Basic statistics for each column of original Numeric Table
     */
    class BasicStatisticsDataCollection : public algorithms::Argument
    {
    public:
        BasicStatisticsDataCollection() : algorithms::Argument(4) {}

        services::SharedPtr<NumericTable> get(BasicStatisticsId id)
        {
            return services::staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
        }

        void set(BasicStatisticsId id, const services::SharedPtr<NumericTable> & value) { Argument::set(id, value); }
    };

    BasicStatisticsDataCollection basicStatistics; /** Basic statistics container */

protected:
    NumericTableDictionaryPtr _ddict;

    size_t _obsnum;

    MemoryStatus _memStatus;
    StorageLayout _layout;

    NormalizationType _normalizationFlag;

    services::Status _status;

protected:
    NumericTable(NumericTableDictionaryPtr ddict, services::Status & /*st*/)
        : _ddict(ddict), _obsnum(0), _memStatus(notAllocated), _layout(layout_unknown), _normalizationFlag(NumericTable::nonNormalized)
    {}

    NumericTable(size_t featnum, size_t obsnum, DictionaryIface::FeaturesEqual featuresEqual, services::Status & st)
        : _obsnum(obsnum), _memStatus(notAllocated), _layout(layout_unknown), _normalizationFlag(NumericTable::nonNormalized)
    {
        _ddict = NumericTableDictionary::create(featnum, featuresEqual, &st);
        if (!st) return;
    }

    virtual services::Status setNumberOfColumnsImpl(size_t ncol) { return _ddict->setNumberOfFeatures(ncol); }

    virtual services::Status setNumberOfRowsImpl(size_t nrow)
    {
        _obsnum = nrow;
        return services::Status();
    }

    virtual services::Status allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) { return services::Status(); }

    virtual void freeDataMemoryImpl() {}

    template <typename DataType>
    DataType getValueImpl(size_t column, size_t row, services::Status & status) const
    {
        const DataType defaultValue = 0;
        if (!status) return defaultValue;
        BlockDescriptor<DataType> bd;
        status |= const_cast<NumericTable *>(this)->getBlockOfColumnValues(column, row, 1, readOnly, bd);
        if (!status) return defaultValue;
        const DataType v = *(bd.getBlockPtr());
        status |= const_cast<NumericTable *>(this)->releaseBlockOfColumnValues(bd);
        return v;
    }

    virtual float getFloatValueImpl(size_t column, size_t row, services::Status & status) const { return getValueImpl<float>(column, row, status); }

    virtual double getDoubleValueImpl(size_t column, size_t row, services::Status & status) const
    {
        return getValueImpl<double>(column, row, status);
    }

    virtual int getIntValueImpl(size_t column, size_t row, services::Status & status) const { return getValueImpl<int>(column, row, status); }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->setSharedPtrObj(_ddict);

        arch->set(_obsnum);

        if (onDeserialize)
        {
            _memStatus = notAllocated;
        }

        arch->set(_layout);

        return services::Status();
    }

private:
    template <typename T>
    services::Status assignImpl(T value)
    {
        size_t nRows = getNumberOfRows();
        size_t nCols = getNumberOfColumns();
        BlockDescriptor<T> block;
        DAAL_CHECK(getBlockOfRows(0, nRows, writeOnly, block), services::ErrorMemoryAllocationFailed)
        T * array = block.getBlockSharedPtr().get();
        for (size_t i = 0; i < nCols * nRows; i++)
        {
            array[i] = value;
        }
        releaseBlockOfRows(block);
        return services::Status();
    }
};
typedef services::SharedPtr<NumericTable> NumericTablePtr;
typedef services::SharedPtr<const NumericTable> NumericTableConstPtr;

template <>
inline float NumericTable::getValue<float>(size_t column, size_t row) const
{
    services::Status status;
    return getFloatValueImpl(column, row, status);
}

template <>
inline double NumericTable::getValue<double>(size_t column, size_t row) const
{
    services::Status status;
    return getDoubleValueImpl(column, row, status);
}

template <>
inline int NumericTable::getValue<int>(size_t column, size_t row) const
{
    services::Status status;
    return getIntValueImpl(column, row, status);
}

template <>
inline float NumericTable::getValue<float>(size_t column, size_t row, services::Status & status) const
{
    return getFloatValueImpl(column, row, status);
}

template <>
inline double NumericTable::getValue<double>(size_t column, size_t row, services::Status & status) const
{
    return getDoubleValueImpl(column, row, status);
}

template <>
inline int NumericTable::getValue<int>(size_t column, size_t row, services::Status & status) const
{
    return getIntValueImpl(column, row, status);
}

/** @} */

} // namespace interface1
using interface1::DenseNumericTableIface;
using interface1::NumericTable;
using interface1::NumericTablePtr;
using interface1::NumericTableConstPtr;

/**
 * Checks the correctness of this numeric table
 * \param[in]  nt                  The numeric table to check
 * \param[in]  description         Additional information about error
 * \param[in]  unexpectedLayouts   The bit mask of invalid layouts for this numeric table.
 * \param[in]  expectedLayouts     The bit mask of valid layouts for this numeric table.
 * \param[in]  nColumns            Required number of columns.
 *                                 nColumns = 0 means that required number of columns is not specified.
 * \param[in]  nRows               Required number of rows.
 *                                 nRows = 0 means that required number of rows is not specified.
 * \param[in]  checkDataAllocation Flag that specifies whether to check the data allocation status
 * \return                         Check status: True if the table satisfies the requirements, false otherwise.
 */
DAAL_EXPORT services::Status checkNumericTable(const NumericTable * nt, const char * description, const int unexpectedLayouts = 0,
                                               const int expectedLayouts = 0, size_t nColumns = 0, size_t nRows = 0, bool checkDataAllocation = true);
/**
 * Converts numeric table with arbitrary storage layout to homogen numeric table of the given type
 * \param[in]  src               Pointer to numeric table
 * \param[in]  type              Type of result numeric table memory
 * \return                       Pointer to homogen numeric table
 */
template <typename DataType>
DAAL_EXPORT daal::data_management::NumericTablePtr convertToHomogen(NumericTable & src, daal::MemType type = daal::dram);
} // namespace data_management
} // namespace daal
#endif
