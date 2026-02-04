/* file: homogen_numeric_table.h */
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
//  Implementation of a homogeneous numeric table.
//--
*/

#ifndef __HOMOGEN_NUMERIC_TABLE_H__
#define __HOMOGEN_NUMERIC_TABLE_H__

#include "services/daal_memory.h"
#include "services/daal_defines.h"

#include "data_management/data/numeric_table.h"
#include "data_management/data/data_serialize.h"
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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__HOMOGENNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a contiguous array
 *  of homogeneous feature vectors. Table rows contain feature vectors,
 *  and columns contain values of individual features.
 *  \tparam DataType Defines the underlying data type that describes a Numeric Table
 */
template <typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT HomogenNumericTable : public NumericTable
{
public:
    DECLARE_SERIALIZABLE_TAG()
    DECLARE_SERIALIZABLE_IMPL()

    DAAL_CAST_OPERATOR(HomogenNumericTable)
    /**
     *  Typedef that stores a datatype used for template instantiation
     */
    typedef DataType baseDataType;

public:
    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddict   Pointer to the predefined NumericTableDictionary
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED HomogenNumericTable(NumericTableDictionary * ddict) : NumericTable(ddict) { _layout = aos; }

    /**
     *  Constructor for an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddictForHomogenNumericTable   Pointer to the predefined NumericTableDictionary
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(NumericTableDictionaryPtr ddictForHomogenNumericTable) : NumericTable(ddictForHomogenNumericTable) { _layout = aos; }

    /**
     *  Constructs an empty Numeric Table with a predefined NumericTableDictionary
     *  \param[in]  ddictForHomogenNumericTable   Pointer to the predefined NumericTableDictionary
     *  \param[out] stat                          Status of the numeric table construction
     *  \return     Empty numeric table with a predefined NumericTableDictionary
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(NumericTableDictionaryPtr ddictForHomogenNumericTable,
                                                                      services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTable, DataType, ddictForHomogenNumericTable);
    }

    /**
     *  Constructor for an empty Numeric Table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable() : NumericTable(0, 0) {}

    /**
     *  Constructs an empty Numeric Table
     *  \param[out] stat    Status of the numeric table construction
     *  \return     Empty numeric table
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL(HomogenNumericTable, DataType);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DataType * const ptr, size_t nColumns = 0, size_t nRows = 0) : NumericTable(nColumns, nRows)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DataType * const ptr, size_t nColumns = 0, size_t nRows = 0,
                                                                      services::Status * stat = NULL)
    {
        return create(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows, stat);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows) : NumericTable(nColumns, nRows)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows,
                                                                      services::Status * stat = NULL)
    {
        return create(DictionaryIface::notEqual, ptr, nColumns, nRows, stat);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, DataType * const ptr = 0, size_t nColumns = 0, size_t nRows = 0)
        : NumericTable(nColumns, nRows, featuresEqual)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual, DataType * const ptr = 0,
                                                                      size_t nColumns = 0, size_t nRows = 0, services::Status * stat = NULL)
    {
        return create(featuresEqual, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows, stat);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows)
        : NumericTable(nColumns, nRows, featuresEqual)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                                      const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows,
                                                                      services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTable, DataType, featuresEqual, ptr, nColumns, nRows);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DataType * const ptr, size_t nColumns, size_t nRows, const DataType & constValue) : NumericTable(nColumns, nRows)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
        this->_status |= assign<DataType>(constValue);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory and initialized with a constant
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DataType * const ptr, size_t nColumns, size_t nRows,
                                                                      const DataType & constValue, services::Status * stat = NULL)
    {
        return create(services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows, constValue, stat);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows, const DataType & constValue)
        : NumericTable(nColumns, nRows)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
        this->_status |= assign<DataType>(constValue);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory and initialized with a constant
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows,
                                                                      const DataType & constValue, services::Status * stat = NULL)
    {
        return create(DictionaryIface::notEqual, ptr, nColumns, nRows, constValue, stat);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, DataType * const ptr, size_t nColumns, size_t nRows,
                        const DataType & constValue)
        : NumericTable(nColumns, nRows, featuresEqual)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
        this->_status |= assign<DataType>(constValue);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory and initialized with a constant
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual, DataType * const ptr,
                                                                      size_t nColumns, size_t nRows, const DataType & constValue,
                                                                      services::Status * stat = NULL)
    {
        return create(featuresEqual, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows, constValue, stat);
    }

    /**
     *  Constructor for a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows,
                        const DataType & constValue)
        : NumericTable(nColumns, nRows, featuresEqual)
    {
        _layout = aos;
        this->_status |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);
        this->_status |= assign<DataType>(constValue);
    }

    /**
     *  Constructs a Numeric Table with user-allocated memory and filling the table with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the numeric table construction
     *  \return     Numeric table with user-allocated memory and initialized with a constant
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                                      const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows,
                                                                      const DataType & constValue, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTable, DataType, featuresEqual, ptr, nColumns, nRows, constValue);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(size_t nColumns, size_t nRows, AllocationFlag memoryAllocationFlag) : NumericTable(nColumns, nRows)
    {
        _layout = aos;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) this->_status |= allocateDataMemoryImpl();
    }

    /**
     *  Constructs a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                    Status of the numeric table construction
     *  \return     Numeric table
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(size_t nColumns, size_t nRows, AllocationFlag memoryAllocationFlag,
                                                                      services::Status * stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, memoryAllocationFlag, stat);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  featuresEqual           Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, AllocationFlag memoryAllocationFlag)
        : NumericTable(nColumns, nRows, featuresEqual)
    {
        _layout = aos;

        NumericTableFeature df;
        df.setType<DataType>();
        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) this->_status |= allocateDataMemoryImpl();
    }

    /**
     *  Constructs a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  featuresEqual           Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                    Status of the numeric table construction
     *  \return     Numeric table
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
                                                                      AllocationFlag memoryAllocationFlag, services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTable, DataType, featuresEqual, nColumns, nRows, memoryAllocationFlag);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue)
        : NumericTable(nColumns, nRows)
    {
        _layout = aos;

        NumericTableFeature df;
        df.setType<DataType>();

        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) this->_status |= allocateDataMemoryImpl();

        this->_status |= assign<DataType>(constValue);
    }

    /**
     *  Constructs a Numeric Table with memory allocation controlled via a flag and fills the table with a constant
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat                    Status of the numeric table construction
     *  \return     Numeric table initialized with a constant
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(size_t nColumns, size_t nRows, AllocationFlag memoryAllocationFlag,
                                                                      const DataType & constValue, services::Status * stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, memoryAllocationFlag, constValue, stat);
    }

    /**
     *  Constructor for a numeric table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  featuresEqual           Flag that makes all features in the numeric table data dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     *  \DAAL_DEPRECATED_USE{ HomogenNumericTable::create }
     */
    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
                        NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue)
        : NumericTable(nColumns, nRows, featuresEqual)
    {
        _layout = aos;

        NumericTableFeature df;
        df.setType<DataType>();

        this->_status |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate)
        {
            this->_status |= allocateDataMemoryImpl();
        }

        this->_status |= assign<DataType>(constValue);
    }

    /**
     *  Constructs a numeric table with memory allocation controlled via a flag and fills the table with a constant
     *  \param[in]  featuresEqual           Flag that makes all features in the numeric table data dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat                    Status of the numeric table construction
     *  \return     Numeric table initialized with a constant
     */
    static services::SharedPtr<HomogenNumericTable<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
                                                                      AllocationFlag memoryAllocationFlag, const DataType & constValue,
                                                                      services::Status * stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(HomogenNumericTable, DataType, featuresEqual, nColumns, nRows, memoryAllocationFlag, constValue);
    }

    virtual ~HomogenNumericTable() { freeDataMemoryImpl(); }

    /**
     *  Returns a pointer to a data set registered in a homogeneous Numeric Table
     *  \return Pointer to the data set
     */
    DataType * getArray() const { return (DataType *)_ptr.get(); }

    /**
     *  Returns a pointer to a data set registered in a homogeneous Numeric Table
     *  \return Pointer to the data set
     */
    services::SharedPtr<DataType> getArraySharedPtr() const { return services::reinterpretPointerCast<DataType, byte>(_ptr); }

    /**
     *  Sets a pointer to a homogeneous data set
     *  \param[in] ptr Pointer to the data set in the homogeneous format
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status setArray(DataType * const ptr)
    {
        freeDataMemoryImpl();

        _ptr = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());

        if (_ptr)
        {
            _memStatus = userAllocated;
        }
        else
        {
            _memStatus = notAllocated;
        }
        return services::Status();
    }

    /**
     *  Sets a pointer to a homogeneous data set
     *  \param[in] ptr Pointer to the data set in the homogeneous format
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status setArray(const services::SharedPtr<DataType> & ptr)
    {
        freeDataMemoryImpl();

        _ptr = services::reinterpretPointerCast<byte, DataType>(ptr);

        if (_ptr)
        {
            _memStatus = userAllocated;
        }
        else
        {
            _memStatus = notAllocated;
        }
        return services::Status();
    }

    /**
     *  Sets a pointer to a homogeneous data set
     *  \param[in] ptr Pointer to the data set in the homogeneous format
     *  \param[in] nRows The number of rows stored in array
     */
    services::Status setArray(DataType * const ptr, size_t nRows)
    {
        freeDataMemoryImpl();

        _ptr    = services::SharedPtr<byte>((byte *)ptr, services::EmptyDeleter());
        _obsnum = nRows;

        if (_ptr)
        {
            _memStatus = userAllocated;
        }
        else
        {
            _memStatus = notAllocated;
        }
        return services::Status();
    }

    /**
     *  Sets a pointer to a homogeneous data set
     *  \param[in] ptr   Pointer to the data set in the homogeneous format
     *  \param[in] nRows The number of rows stored in array
     */
    services::Status setArray(services::SharedPtr<DataType> ptr, size_t nRows)
    {
        freeDataMemoryImpl();

        _ptr    = services::reinterpretPointerCast<byte, DataType>(ptr);
        _obsnum = nRows;

        if (_ptr)
        {
            _memStatus = userAllocated;
        }
        else
        {
            _memStatus = notAllocated;
        }
        return services::Status();
    }

    /**
     *  Fills a numeric table with a constant
     *  \param[in]  value  Constant to initialize entries of the homogeneous numeric table
     */
    template <typename T>
    DAAL_EXPORT services::Status assign(T value)
    {
        if (_memStatus == notAllocated) return services::Status(services::ErrorEmptyHomogenNumericTable);

        size_t nColumns = getNumberOfColumns();
        size_t nRows    = getNumberOfRows();

        DataType * ptr         = (DataType *)_ptr.get();
        DataType valueDataType = (DataType)value;

        internal::vectorAssignValueToArray<DataType>(ptr, nColumns * nRows, valueDataType);

        return services::Status();
    }

    /**
     *  Returns a pointer to the i-th row of a data set
     *  \param[in]  i  Index of the row
     *  \return Pointer to the i-th row
     */
    DataType * operator[](size_t i)
    {
        size_t nColumns = getNumberOfColumns();
        return (DataType *)_ptr.get() + i * nColumns;
    }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(float value) override { return assign<DataType>((DataType)value); }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(double value) override { return assign<DataType>((DataType)value); }

    /**
     * \copydoc NumericTable::assign
     */
    virtual services::Status assign(int value) override { return assign<DataType>((DataType)value); }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) override
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) override
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) override
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) override { return releaseTBlock<double>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) override { return releaseTBlock<float>(block); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) override { return releaseTBlock<int>(block); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) override
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) override
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) override
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) override { return releaseTFeature<double>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) override { return releaseTFeature<float>(block); }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) override { return releaseTFeature<int>(block); }

protected:
    services::SharedPtr<byte> _ptr;

    HomogenNumericTable(services::Status & st) : NumericTable(0, 0, DictionaryIface::notEqual, st) {}

    HomogenNumericTable(NumericTableDictionaryPtr ddictForHomogenNumericTable, services::Status & st) : NumericTable(ddictForHomogenNumericTable, st)
    {
        _layout = aos;
    }

    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows,
                        services::Status & st)
        : NumericTable(nColumns, nRows, featuresEqual, st)
    {
        _layout = aos;
        st |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);
    }

    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, const services::SharedPtr<DataType> & ptr, size_t nColumns, size_t nRows,
                        const DataType & constValue, services::Status & st)
        : NumericTable(nColumns, nRows, featuresEqual, st)
    {
        _layout = aos;
        st |= setArray(ptr, nRows);

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);
        st |= assign<DataType>(constValue);
    }

    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, AllocationFlag memoryAllocationFlag,
                        services::Status & st)
        : NumericTable(nColumns, nRows, featuresEqual, st)
    {
        _layout = aos;

        NumericTableFeature df;
        df.setType<DataType>();
        st |= _ddict->setAllFeatures(df);

        if (memoryAllocationFlag == doAllocate) st |= allocateDataMemoryImpl();
    }

    HomogenNumericTable(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
                        NumericTable::AllocationFlag memoryAllocationFlag, const DataType & constValue, services::Status & st);

    services::Status allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) override
    {
        freeDataMemoryImpl();

        size_t size = getNumberOfColumns() * getNumberOfRows();

        if (!(0 == getNumberOfColumns()) && !(0 == getNumberOfRows()))
        {
            DAAL_CHECK((getNumberOfColumns() == size / getNumberOfRows()),
                       services::throwIfPossible(services::Status(services::ErrorBufferSizeIntegerOverflow)));

            size_t sizeEx = size * sizeof(DataType);
            DAAL_CHECK((size == sizeEx / sizeof(DataType)), services::throwIfPossible(services::Status(services::ErrorBufferSizeIntegerOverflow)));
        }

        if (size == 0)
        {
            return services::Status(getNumberOfColumns() == 0 ? services::ErrorIncorrectNumberOfFeatures :
                                                                services::ErrorIncorrectNumberOfObservations);
        }

        _ptr = services::SharedPtr<byte>((byte *)daal::services::daal_malloc(size * sizeof(DataType)), services::ServiceDeleter());

        if (!_ptr) return services::Status(services::ErrorMemoryAllocationFailed);

        _memStatus = internallyAllocated;
        return services::Status();
    }

    void freeDataMemoryImpl() override
    {
        _ptr       = services::SharedPtr<byte>();
        _memStatus = notAllocated;
    }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * archive)
    {
        NumericTable::serialImpl<Archive, onDeserialize>(archive);

        if (onDeserialize)
        {
            allocateDataMemoryImpl();
        }

        size_t size = getNumberOfColumns() * getNumberOfRows();

        archive->set((DataType *)_ptr.get(), size);

        return services::Status();
    }

private:
    byte * internal_getBlockOfRows(size_t idx)
    {
        size_t _featnum = _ddict->getNumberOfFeatures();
        return _ptr.get() + _featnum * idx * sizeof(DataType);
    }

    byte * internal_getBlockOfRows(size_t idx, size_t feat_idx)
    {
        size_t _featnum = _ddict->getNumberOfFeatures();
        return _ptr.get() + _featnum * idx * sizeof(DataType) + feat_idx * sizeof(DataType);
    }

protected:
    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(0, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(ncols, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if (IsSameType<T, DataType>::value)
        {
            block.setPtr(&_ptr, internal_getBlockOfRows(idx), ncols, nrows);
        }
        else
        {
            if (!block.resizeBuffer(ncols, nrows)) return services::Status(services::ErrorMemoryAllocationFailed);

            if (rwFlag & (int)readOnly)
            {
                byte * location = internal_getBlockOfRows(idx);

                for (size_t i = 0; i < nrows; i++)
                {
                    internal::getVectorUpCast(features::internal::getIndexNumType<DataType>(), internal::getConversionDataType<T>())(
                        ncols, ((DataType *)location) + i * ncols, ((T *)block.getBlockPtr()) + i * ncols);
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
            byte * location = internal_getBlockOfRows(block.getRowsOffset());
            size_t ncols    = getNumberOfColumns();
            size_t nrows    = block.getNumberOfRows();

            if (IsSameType<T, DataType>::value)
            {
                if ((T *)block.getBlockPtr() != (T *)location)
                {
                    int result =
                        daal::services::internal::daal_memcpy_s(location, nrows * ncols * sizeof(T), block.getBlockPtr(), nrows * ncols * sizeof(T));
                    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
                }
            }
            else
            {
                for (size_t i = 0; i < nrows; i++)
                {
                    internal::getVectorDownCast(features::internal::getIndexNumType<DataType>(), internal::getConversionDataType<T>())(
                        ncols, ((T *)block.getBlockPtr()) + i * ncols, ((DataType *)location) + i * ncols);
                }
            }
        }
        block.reset();
        return services::Status();
    }

    template <typename T>
    services::Status getTFeature(size_t feat_idx, size_t idx, size_t nrows, int rwFlag, BlockDescriptor<T> & block)
    {
        size_t ncols = getNumberOfColumns();
        size_t nobs  = getNumberOfRows();
        block.setDetails(feat_idx, idx, rwFlag);

        if (idx >= nobs)
        {
            block.resizeBuffer(1, 0);
            return services::Status();
        }

        nrows = (idx + nrows < nobs) ? nrows : nobs - idx;

        if ((IsSameType<T, DataType>::value) && (ncols == 1))
        {
            block.setPtr(&_ptr, internal_getBlockOfRows(idx), ncols, nrows);
        }
        else
        {
            if (!block.resizeBuffer(1, nrows)) return services::Status(services::ErrorMemoryAllocationFailed);

            if (rwFlag & (int)readOnly)
            {
                DataType * location = (DataType *)internal_getBlockOfRows(idx, feat_idx);
                T * buffer          = block.getBlockPtr();
                internal::getVectorStrideUpCast(features::internal::getIndexNumType<DataType>(), internal::getConversionDataType<T>())(
                    nrows, location, sizeof(DataType) * ncols, buffer, sizeof(T));
            }
        }
        return services::Status();
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block)
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            size_t ncols        = getNumberOfColumns();
            DataType * location = (DataType *)internal_getBlockOfRows(block.getRowsOffset(), block.getColumnsOffset());
            internal::getVectorStrideDownCast(features::internal::getIndexNumType<DataType>(), internal::getConversionDataType<T>())(
                block.getNumberOfRows(), block.getBlockPtr(), sizeof(T), location, ncols * sizeof(DataType));
        }
        block.reset();
        return services::Status();
    }

    services::Status setNumberOfColumnsImpl(size_t ncol) override
    {
        if (_ddict->getNumberOfFeatures() != ncol)
        {
            _ddict->resetDictionary();
            _ddict->setNumberOfFeatures(ncol);

            NumericTableFeature df;
            df.setType<DataType>();
            _ddict->setAllFeatures(df);
        }
        return services::Status();
    }
};
/** @} */
} // namespace interface1
using interface1::HomogenNumericTable;

} // namespace data_management
} // namespace daal
#endif
