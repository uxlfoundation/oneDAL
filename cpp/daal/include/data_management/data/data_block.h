/* file: data_block.h */
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
//  Implementation of the DataBlock type
//--
*/

#ifndef __DAAL_DATABLOCK_H__
#define __DAAL_DATABLOCK_H__

#include "services/base.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup serialization
 * @{
 */
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATABLOCKIFACE"></a>
 * \brief Abstract interface class for a data management component responsible for a pointer to a byte array and its size.
 * This class declares the most general methods for data access.
 */
class DataBlockIface : public Base
{
public:
    ~DataBlockIface() override {}
    /**
    * Returns a pointer to a byte array stored in DataBlock
    * \return Pointer to the byte array stored in DataBlock
    */
    virtual byte * getPtr() const = 0;
    /**
    * Returns a pointer to a byte array stored in DataBlock
    * \return Pointer to the byte array stored in DataBlock
    */
    virtual services::SharedPtr<byte> getSharedPtr() const = 0;
    /**
     * Returns the size of a byte array stored in DataBlock
     * \return Size of the byte array stored in DataBlock
     */
    virtual size_t getSize() const = 0;
    /**
     * Sets a pointer to a byte array
     * \param[in] ptr Pointer to the byte array
     */
    virtual void setPtr(byte * ptr) = 0;
    /**
     * Sets a pointer to a byte array
     * \param[in] ptr Pointer to the byte array
     */
    virtual void setPtr(const services::SharedPtr<byte> & ptr) = 0;
    /**
     * Sets the size of a byte array
     * \param[in] size Size of the byte array
     */
    virtual void setSize(size_t size) = 0;
};
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATABLOCK"></a>
 * \brief Class that stores a pointer to a byte array and its size. Not responsible for memory management
 */
class DAAL_EXPORT DataBlock : public DataBlockIface
{
public:
    /**
     * Default constructor. Creates an empty DataBlock of zero size with a zero pointer to a byte array
     */
    DataBlock() : _ptr(), _size(0) {}
    /**
     * Constructor. Creates DataBlock with a user-defined byte array
     * \param ptr Pointer to the byte array
     * \param size Size of the byte array
     */
    DataBlock(byte * ptr, size_t size) : _ptr(ptr, services::EmptyDeleter()), _size(size) {}
    /**
     * Constructor. Creates DataBlock with a user-defined byte array
     * \param ptr Pointer to the byte array
     * \param size Size of the byte array
     */
    DataBlock(const services::SharedPtr<byte> & ptr, size_t size) : _ptr(ptr), _size(size) {}
    /**
     * Constructor. Creates an empty DataBlock of a predefined size
     * \param size Size of the byte array
     */
    DataBlock(size_t size) : _ptr(), _size(size) {}
    /**
     * Copy constructor. Copies a pointer and the size stored in another DataBlock
     * \param block Reference to DataBlock
     */
    DataBlock(const DataBlock & block) : _ptr(block._ptr), _size(block._size) {}

    ~DataBlock() override {}

    virtual byte * getPtr() const override { return _ptr.get(); }

    virtual services::SharedPtr<byte> getSharedPtr() const override { return _ptr; }

    virtual size_t getSize() const override { return _size; }

    virtual void setPtr(byte * ptr) override { _ptr = services::SharedPtr<byte>(ptr, services::EmptyDeleter()); }

    virtual void setPtr(const services::SharedPtr<byte> & ptr) override { _ptr = ptr; }

    virtual void setSize(size_t size) override { _size = size; }

private:
    services::SharedPtr<byte> _ptr;
    size_t _size;
};
typedef services::SharedPtr<DataBlock> DataBlockPtr;
/** @} */
} // namespace interface1
using interface1::DataBlock;
using interface1::DataBlockPtr;
using interface1::DataBlockIface;
} // namespace data_management
} // namespace daal

#endif
