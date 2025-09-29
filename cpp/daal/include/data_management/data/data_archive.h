/* file: data_archive.h */
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
//  Declaration and implementation of classes that support serialization and deserialization methods
//--
*/

#ifndef __DATA_ARCHIVE_H__
#define __DATA_ARCHIVE_H__

#include "services/base.h"
#include "services/library_version_info.h"
#include "services/daal_memory.h"
#include "services/collection.h"

#include "data_management/data/data_block.h"
#include "data_management/data/factory.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_collection.h"
#include "data_management/features/defines.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATAARCHIVEIFACE"></a>
 *  \brief Abstract interface class that defines methods to access and modify a serialized object.
 *  This class declares the most generic access and modification methods.
 */
class DataArchiveIface : public Base
{
public:
    ~DataArchiveIface() DAAL_C11_OVERRIDE {}

    /**
     *  Copies data into an archive
     *  \param[in]  ptr  Pointer to the data represented in the byte format
     *  \param[in]  size Size of the data array
     */
    virtual void write(byte * ptr, size_t size) = 0;

    /**
     *  Copies the content of an archive into a byte array
     *  \param[in]  ptr  Pointer to the array that represents the data
     *  \param[in]  size Size of the data array
     */
    virtual void read(byte * ptr, size_t size) = 0;

    /**
     *  Returns the size of an archive
     *  \return Size of the archive in bytes
     */
    virtual size_t getSizeOfArchive() const = 0;

    /**
     *  Returns a data archive in the byte format
     *  \return Pointer to the byte buffer with the archive data
     */
    virtual services::SharedPtr<byte> getArchiveAsArraySharedPtr() const = 0;

    /**
     *  Returns a data archive in the byte format
     *  \return Pointer to the byte buffer with the archive data
     *  \DAAL_DEPRECATED_USE{DataArchiveIface::getArchiveAsArraySharedPtr}
     */
    DAAL_DEPRECATED_VIRTUAL virtual byte * getArchiveAsArray() { return NULL; }

    /**
     *  Returns a data archive in the STL string format
     *  \return Object of the std::string type with the archive data
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual std::string getArchiveAsString() = 0;

    /**
     *  Copies a data archive in the byte format to user-specified memory
     *  \param[in]  ptr     Pointer to the byte array
     *  \param[in]  maxLength Size of the array
     *  \return Actual size of the data archive in bytes
     */
    virtual size_t copyArchiveToArray(byte * ptr, size_t maxLength) const = 0;

    /**
     * Sets the major version of the archive
     * \param[in] majorVersion The major version of the archive
     */
    virtual void setMajorVersion(int majorVersion) = 0;

    /**
     * Sets the minor version of the archive
     * \param[in] minorVersion The minor version of the archive
     */
    virtual void setMinorVersion(int minorVersion) = 0;

    /**
     * Sets the update version of the archive
     * \param[in] updateVersion The update version of the archive
     */
    virtual void setUpdateVersion(int updateVersion) = 0;

    /**
     * Returns the major version of the archive
     * \return The major version of the archive
     */
    virtual int getMajorVersion() = 0;

    /**
     * Returns the minor version of the archive
     * \return The minor version of the archive
     */
    virtual int getMinorVersion() = 0;

    /**
     * Returns the update version of the archive
     * \return The update version of the archive
     */
    virtual int getUpdateVersion() = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATAARCHIVEIMPL"></a>
 *  \brief Abstract interface class that defines methods to access and modify a serialized object.
 *  This class implements the most general serialization methods.
 */
class DataArchiveImpl : public DataArchiveIface
{
public:
    DataArchiveImpl() : _majorVersion(0), _minorVersion(0), _updateVersion(0) {}

    ~DataArchiveImpl() DAAL_C11_OVERRIDE {}

    void setMajorVersion(int majorVersion) DAAL_C11_OVERRIDE { _majorVersion = majorVersion; }

    void setMinorVersion(int minorVersion) DAAL_C11_OVERRIDE { _minorVersion = minorVersion; }

    void setUpdateVersion(int updateVersion) DAAL_C11_OVERRIDE { _updateVersion = updateVersion; }

    int getMajorVersion() DAAL_C11_OVERRIDE { return _majorVersion; }

    int getMinorVersion() DAAL_C11_OVERRIDE { return _minorVersion; }

    int getUpdateVersion() DAAL_C11_OVERRIDE { return _updateVersion; }

    virtual services::SharedPtr<services::ErrorCollection> getErrors() = 0;

protected:
    int _majorVersion;
    int _minorVersion;
    int _updateVersion;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATAARCHIVE"></a>
 *  \brief Implements the abstract DataArchiveIface interface
 */
class DataArchive : public DataArchiveImpl
{
public:
    /**
     *  Constructor of an empty data archive
     */
    DataArchive() : _errors(new services::ErrorCollection()), minBlocksNum(16), minBlockSize(1024 * 16)
    {
        blockPtr           = 0;
        blockAllocatedSize = 0;
        blockOffset        = 0;
        arraysSize         = 0;
        currentWriteBlock  = -1;

        currentReadBlock       = 0;
        currentReadBlockOffset = 0;

        serializedBuffer = 0;

        addBlock(minBlockSize);
    }

    /**
     *  Copy constructor of a data archive
     */
    DataArchive(const DataArchive & arch) : minBlocksNum(16), minBlockSize(1024 * 16)
    {
        blockPtr           = 0;
        blockAllocatedSize = 0;
        blockOffset        = 0;
        arraysSize         = 0;
        currentWriteBlock  = -1;

        currentReadBlock       = 0;
        currentReadBlockOffset = 0;

        serializedBuffer = 0;

        size_t size = arch.getSizeOfArchive();
        addBlock(size);
        arch.copyArchiveToArray(blockPtr[currentWriteBlock], size);

        blockOffset[currentWriteBlock] += size;
    }

    /**
     *  Constructor of a data archive from data in a byte array
     *  \param[in]  ptr  Pointer to the array that represents the data
     *  \param[in]  size Size of the data array
     */
    DataArchive(byte * ptr, size_t size) : _errors(new services::ErrorCollection()), minBlocksNum(16), minBlockSize(1024 * 16)
    {
        blockPtr           = 0;
        blockAllocatedSize = 0;
        blockOffset        = 0;
        arraysSize         = 0;
        currentWriteBlock  = -1;

        currentReadBlock       = 0;
        currentReadBlockOffset = 0;

        serializedBuffer = 0;

        addBlock(size);

        int result = daal::services::internal::daal_memcpy_s(blockPtr[currentWriteBlock], size, ptr, size);
        if (result)
        {
            this->_errors->add(services::ErrorMemoryCopyFailedInternal);
        }

        blockOffset[currentWriteBlock] += size;
    }

    ~DataArchive() DAAL_C11_OVERRIDE
    {
        for (int i = 0; i <= currentWriteBlock; i++)
        {
            daal::services::daal_free(blockPtr[i]);
            blockPtr[i] = NULL;
        }
        daal::services::daal_free(blockPtr);
        daal::services::daal_free(blockAllocatedSize);
        daal::services::daal_free(blockOffset);
        if (serializedBuffer)
        {
            daal::services::daal_free(serializedBuffer);
        }

        blockPtr           = NULL;
        blockAllocatedSize = NULL;
        blockOffset        = NULL;
        serializedBuffer   = NULL;
    }

    void write(byte * ptr, size_t size) DAAL_C11_OVERRIDE
    {
        size_t alignedSize = alignValueUp(size);
        if (blockAllocatedSize[currentWriteBlock] < blockOffset[currentWriteBlock] + alignedSize)
        {
            addBlock(alignedSize);
        }

        size_t offset = blockOffset[currentWriteBlock];

        int result = daal::services::internal::daal_memcpy_s(&(blockPtr[currentWriteBlock][offset]), alignedSize, ptr, size);
        if (result)
        {
            this->_errors->add(services::ErrorMemoryCopyFailedInternal);
            return;
        }
        for (size_t i = size; i < alignedSize; i++)
        {
            blockPtr[currentWriteBlock][offset + i] = 0;
        }

        blockOffset[currentWriteBlock] += alignedSize;
    }

    void read(byte * ptr, size_t size) DAAL_C11_OVERRIDE
    {
        size_t alignedSize = alignValueUp(size);
        if (blockOffset[currentReadBlock] < currentReadBlockOffset + alignedSize)
        {
            this->_errors->add(services::ErrorDataArchiveInternal);
            return;
        }

        int result = daal::services::internal::daal_memcpy_s(ptr, size, &(blockPtr[currentReadBlock][currentReadBlockOffset]), size);
        if (result)
        {
            this->_errors->add(services::ErrorMemoryCopyFailedInternal);
            return;
        }

        currentReadBlockOffset += alignedSize;
        if (blockOffset[currentReadBlock] == currentReadBlockOffset)
        {
            currentReadBlock++;
            currentReadBlockOffset = 0;
        }
    }

    size_t getSizeOfArchive() const DAAL_C11_OVERRIDE
    {
        size_t size = 0;
        for (int i = 0; i <= currentWriteBlock; i++)
        {
            size += blockOffset[i];
        }
        return size;
    }

    services::SharedPtr<byte> getArchiveAsArraySharedPtr() const DAAL_C11_OVERRIDE
    {
        size_t length = getSizeOfArchive();

        if (length == 0)
        {
            return services::SharedPtr<byte>();
        }

        services::SharedPtr<byte> serializedBufferPtr((byte *)daal::services::daal_malloc(length), services::ServiceDeleter());
        if (!serializedBufferPtr)
        {
            return services::SharedPtr<byte>();
        }

        copyArchiveToArray(serializedBufferPtr.get(), length);

        return serializedBufferPtr;
    }

    byte * getArchiveAsArray() DAAL_C11_OVERRIDE
    {
        if (serializedBuffer)
        {
            return serializedBuffer;
        }

        size_t length = getSizeOfArchive();

        if (length == 0)
        {
            return 0;
        }

        serializedBuffer = (byte *)daal::services::daal_malloc(length);
        if (serializedBuffer == 0)
        {
            return 0;
        }

        copyArchiveToArray(serializedBuffer, length);

        return serializedBuffer;
    }

    std::string getArchiveAsString() DAAL_C11_OVERRIDE
    {
        size_t length = getSizeOfArchive();
        char * buffer = (char *)getArchiveAsArray();

        return std::string(buffer, length);
    }

    size_t copyArchiveToArray(byte * ptr, size_t maxLength) const DAAL_C11_OVERRIDE
    {
        size_t length = getSizeOfArchive();

        if (length == 0 || length > maxLength)
        {
            return length;
        }

        size_t offset = 0;
        int result    = 0;
        for (int i = 0; i <= currentWriteBlock; i++)
        {
            size_t blockSize = blockOffset[i];

            result |= daal::services::internal::daal_memcpy_s(&(ptr[offset]), blockSize, blockPtr[i], blockSize);

            offset += blockSize;
        }
        if (result)
        {
            this->_errors->add(services::ErrorMemoryCopyFailedInternal);
            return 0;
        }

        return length;
    }

    /**
     * Returns errors during the computation
     * \return Errors during the computation
     */
    services::SharedPtr<services::ErrorCollection> getErrors() DAAL_C11_OVERRIDE { return _errors; }

protected:
    void addBlock(size_t minNewSize)
    {
        if (currentWriteBlock + 1 == arraysSize)
        {
            byte ** oldBlockPtr            = blockPtr;
            size_t * oldBlockAllocatedSize = blockAllocatedSize;
            size_t * oldBlockOffset        = blockOffset;
            int result                     = 0;

            blockPtr           = (byte **)daal::services::daal_malloc(sizeof(byte *) * (arraysSize + minBlocksNum));
            blockAllocatedSize = (size_t *)daal::services::daal_malloc(sizeof(size_t) * (arraysSize + minBlocksNum));
            blockOffset        = (size_t *)daal::services::daal_malloc(sizeof(size_t) * (arraysSize + minBlocksNum));

            if (blockPtr == 0 || blockAllocatedSize == 0 || blockOffset == 0)
            {
                return;
            }

            result |= daal::services::internal::daal_memcpy_s(blockPtr, arraysSize * sizeof(byte *), oldBlockPtr, arraysSize * sizeof(byte *));
            result |= daal::services::internal::daal_memcpy_s(blockAllocatedSize, arraysSize * sizeof(size_t), oldBlockAllocatedSize,
                                                              arraysSize * sizeof(size_t));
            result |= daal::services::internal::daal_memcpy_s(blockOffset, arraysSize * sizeof(size_t), oldBlockOffset, arraysSize * sizeof(size_t));
            if (result)
            {
                this->_errors->add(services::ErrorMemoryCopyFailedInternal);
                return;
            }

            daal::services::daal_free(oldBlockPtr);
            daal::services::daal_free(oldBlockAllocatedSize);
            daal::services::daal_free(oldBlockOffset);

            arraysSize += minBlocksNum;
        }

        currentWriteBlock++;

        size_t allocationSize = (minBlockSize > minNewSize) ? minBlockSize : minNewSize;

        blockPtr[currentWriteBlock]           = (byte *)daal::services::daal_malloc(allocationSize);
        blockAllocatedSize[currentWriteBlock] = allocationSize;
        blockOffset[currentWriteBlock]        = 0;
    }

    inline size_t alignValueUp(size_t value)
    {
        if (_majorVersion == 2016 && _minorVersion == 0 && _updateVersion == 0)
        {
            return value;
        }

        size_t alignm1 = DAAL_MALLOC_DEFAULT_ALIGNMENT - 1;

        size_t alignedValue = value + alignm1;
        alignedValue &= ~alignm1;
        return alignedValue;
    }

    services::SharedPtr<services::ErrorCollection> _errors;

private:
    int minBlocksNum;
    size_t minBlockSize;

    byte ** blockPtr;
    size_t * blockAllocatedSize;
    size_t * blockOffset;

    int arraysSize;

    int currentWriteBlock;

    int currentReadBlock;
    size_t currentReadBlockOffset;

    byte * serializedBuffer;

    DataArchive & operator=(const DataArchive &);
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__INPUTDATAARCHIVE"></a>
 *  \brief Provides methods to create an archive data object (serialized) and access this object
 */
class InputDataArchive : public Base
{
public:
    /**
     *  Default constructor
     */
    InputDataArchive() : _finalized(false), _errors(new services::ErrorCollection())
    {
        _arch = new DataArchive;
        archiveHeader();
    }

    /**
     *  Constructor of an input data archive from a DataArchiveIface
     *  The new InputDataArchive object will own the provided pointer
     *  and free it when it gets deleted.
     */
    InputDataArchive(DataArchiveIface * arch) : _finalized(false), _errors(new services::ErrorCollection())
    {
        _arch = arch;
        archiveHeader();
    }

    ~InputDataArchive() DAAL_C11_OVERRIDE { delete _arch; }

    /**
     *  Generates a header for a data archive
     */
    void archiveHeader()
    {
        int headerValues[8] = { 0x4441414C, __INTEL_DAAL__, __INTEL_DAAL_MINOR__, __INTEL_DAAL_UPDATE__, 0, 0, 0, 0 };

        _arch->setMajorVersion(headerValues[1]);
        _arch->setMinorVersion(headerValues[2]);
        _arch->setUpdateVersion(headerValues[3]);
        for (size_t i = 0; i < 8; i++)
        {
            _arch->write((byte *)&headerValues[i], sizeof(int));
        }
    }

    /**
     *  Generates a footer for a data archive
     */
    void archiveFooter() { _finalized = true; }

    /**
     *  Generates a header for a segment in the DataArchive object
     */
    void segmentHeader(int tag = 0) { _arch->write((byte *)&tag, sizeof(int)); }

    /**
     *  Generates a footer for a segment in the DataArchive object
     */
    void segmentFooter() {}

    /**
     *  Performs data serialization of one value of the basic datatype
     *  \tparam  T        basic datatype
     *  \param[in]   val  Reference to the data to serialize
     */
    template <typename T>
    void set(T & val)
    {
        _arch->write((byte *)&val, sizeof(T));
    }

    /**
     *  Performs data serialization of Collection of the basic datatype
     *  \tparam  T        basic datatype
     *  \param[in]   val  Reference to the data to serialize
     */
    template <typename T>
    void set(daal::services::Collection<T> & val)
    {
        size_t size = val.size();
        _arch->write((byte *)&size, sizeof(size_t));
        for (size_t i = 0; i < size; i++)
        {
            _arch->write((byte *)&(val[i]), sizeof(T));
        }
    }

    /**
     *  Performs data serialization of an array of values of the basic datatype
     *  \tparam  T         Basic datatype
     *  \param[in]   ptr   Pointer to the array of data to convert to the serialized format
     *  \param[in]   size  Size of the array pointed to by ptr
     */
    template <typename T>
    void set(T * ptr, size_t size)
    {
        if (size)
        {
            _arch->write((byte *)ptr, size * sizeof(T));
        }
    }

    /**
     *  Performs data serialization creating a data segment
     *  \tparam  T        Class that implements SerializationIface
     *  \param[in]   ptr  Pointer to an array of data to convert to the serialized format
     *  \param[in]   size Size of the array pointed to by ptr
     */
    template <typename T>
    void setObj(T * ptr, size_t size = 1)
    {
        for (size_t i = 0; i < size; i++)
        {
            ptr[i].serializeImpl(this);
        }
    }

    /**
     *  Performs data serialization creating a data segment
     *  \param[in]   ptr  Pointer to the serializable object
     */
    void setSingleObj(SerializationIface ** ptr)
    {
        int isNull = (*ptr == 0);
        set(isNull);

        if (!isNull)
        {
            (*ptr)->serialize(*this);
        }
    }

    /**
     *  Performs data serialization creating a data segment
     *  \param[in]   obj  Serializable object
     */
    template <typename T>
    void setSharedPtrObj(services::SharedPtr<T> & obj)
    {
        data_management::SerializationIface * ptr = obj.get();
        setSingleObj(&ptr);
    }

    /**
     *  Returns a data archive in the byte format
     *  \return Pointer to the byte buffer with the archive data
     */
    services::SharedPtr<byte> getArchiveAsArraySharedPtr()
    {
        if (!_finalized)
        {
            archiveFooter();
        }
        return _arch->getArchiveAsArraySharedPtr();
    }

    /**
     *  Returns a data archive in the byte format
     *  \return Pointer to the byte buffer with the archive data
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED byte * getArchiveAsArray()
    {
        if (!_finalized)
        {
            archiveFooter();
        }
        return _arch->getArchiveAsArray();
    }

    /**
     *  Returns a data archive in the  byte format
     *  \param[in]   ptr  Pointer to the byte buffer with the archive data
     *  \param[in]   size Pointer to the size of the array
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED void getArchiveAsArray(const byte ** ptr, size_t * size)
    {
        if (!_finalized)
        {
            archiveFooter();
        }

        *ptr  = (byte *)_arch->getArchiveAsArray();
        *size = _arch->getSizeOfArchive();
    }

    /**
     *  Returns the size of an archive
     *  \return Size of the archive in bytes
     */
    size_t getSizeOfArchive()
    {
        if (!_finalized)
        {
            archiveFooter();
        }

        return _arch->getSizeOfArchive();
    }

    /**
     *  Returns a data archive in the STL string format
     *  \return Object of the std::string type with the archive data
     *  \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED std::string getArchiveAsString()
    {
        if (!_finalized)
        {
            archiveFooter();
        }

        return _arch->getArchiveAsString();
    }

    /**
     *  Copies a data archive in the byte format to user-specified memory
     *  \param[in]  ptr     Pointer to the byte array
     *  \param[in]  maxLength Size of the array
     *  \return Actual size of the data archive in bytes
     */
    size_t copyArchiveToArray(byte * ptr, size_t maxLength)
    {
        if (!_finalized)
        {
            archiveFooter();
        }

        return _arch->copyArchiveToArray(ptr, maxLength);
    }

    /**
     *  Returns a data archive object of the InputDataArchive type
     *  \return Data archive object
     */
    const DataArchive & getDataArchive() { return *static_cast<DataArchive *>(_arch); }

    /**
    * Returns errors during the computation
    * \return Errors during the computation
    */
    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        if (_arch)
        {
            services::SharedPtr<services::ErrorCollection> errors = static_cast<DataArchiveImpl *>(_arch)->getErrors();
            if (errors.get())
            {
                _errors->add(*errors);
            }
        }
        return _errors;
    }

protected:
    DataArchiveIface * _arch;
    bool _finalized;
    services::SharedPtr<services::ErrorCollection> _errors;

private:
    InputDataArchive(const InputDataArchive &);
    InputDataArchive & operator=(const InputDataArchive &);
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__OUTPUTDATAARCHIVE"></a>
 *  \brief Provides methods to restore an object from its serialized counterpart and access the restored object
 */
class OutputDataArchive : public Base
{
public:
    /**
     *  Constructor of an output data archive from an input data archive
     */
    OutputDataArchive(InputDataArchive & arch) : _errors(new services::ErrorCollection())
    {
        _arch = new DataArchive(arch.getDataArchive());
        archiveHeader();
    }

    /**
     *  Constructor of an output data archive from a DataArchiveIface
     *  The new OutputDataArchive object will own the provided pointer
     *  and free it when it gets deleted.
     */
    OutputDataArchive(DataArchiveIface * arch) : _errors(new services::ErrorCollection())
    {
        _arch = arch;
        archiveHeader();
    }

    /**
     *  Constructor of an output data archive from a byte array
     */
    OutputDataArchive(byte * ptr, size_t size) : _errors(new services::ErrorCollection())
    {
        _arch = new DataArchive(ptr, size);
        archiveHeader();
    }

    ~OutputDataArchive() DAAL_C11_OVERRIDE { delete _arch; }

    /**
     *  Reads the header from a data archive
     */
    void archiveHeader() const
    {
        int headerValues[8];

        for (size_t i = 0; i < 8; i++)
        {
            _arch->read((byte *)&headerValues[i], sizeof(int));
        }

        _arch->setMajorVersion(headerValues[1]);
        _arch->setMinorVersion(headerValues[2]);
        _arch->setUpdateVersion(headerValues[3]);
    }

    /**
     *  Reads the footer from a data archive
     */
    void archiveFooter() const {}

    /**
     *  Reads the header for a segment from the DataArchive object
     */
    int segmentHeader() const
    {
        int tag = 0;
        _arch->read((byte *)&tag, sizeof(int));
        return tag;
    }

    /**
     *  Reads the footer for a segment from the DataArchive object
     */
    void segmentFooter() const {}

    /**
     *  Performs data deserialization of one value of the basic datatype
     *  \tparam  T        basic datatype
     *  \param[in]   val  Reference to the data to deserialize
     */
    template <typename T>
    void set(T & val) const
    {
        _arch->read((byte *)&val, sizeof(T));
    }

    /**
     *  Performs data deserialization of Collection of the basic datatype
     *  \tparam  T        basic datatype
     *  \param[in]   val  Reference to the data to serialize
     */
    template <typename T>
    void set(daal::services::Collection<T> & val) const
    {
        size_t size = 0;
        _arch->read((byte *)&size, sizeof(size_t));
        val.clear();
        for (size_t i = 0; i < size; i++)
        {
            T v;
            _arch->read((byte *)&v, sizeof(T));
            val.push_back(v);
        }
    }

    /**
     *  Performs data deserialization of an array of values of the basic datatype
     *  \tparam  T         Basic datatype
     *  \param[in]   ptr   Pointer to the array of data to convert from the serialized format
     *  \param[in]   size  Size of the array pointed to by ptr
     */
    template <typename T>
    void set(T * ptr, size_t size) const
    {
        if (size)
        {
            _arch->read((byte *)ptr, size * sizeof(T));
        }
    }

    /**
     *  Performs data deserialization of a data segment
     *  \tparam  T        Class that implements SerializationIface
     *  \param[in]   ptr  Pointer to an array of empty objects of the T class to deserialized data
     *  \param[in]   size Size of the array pointed to by ptr
     */
    template <typename T>
    void setObj(T * ptr, size_t size = 1) const
    {
        for (size_t i = 0; i < size; i++)
        {
            ptr[i].deserializeImpl(this);
        }
    }

    /**
     *  Performs data deserialization creating a data segment
     *  \param[in]   ptr  Pointer to the serializable object
     */
    void setSingleObj(SerializationIface ** ptr) const
    {
        int isNull = 0;
        set(isNull);

        if (isNull)
        {
            *ptr = 0;
            return;
        }

        const int serTag = segmentHeader();

        *ptr = Factory::instance().createObject(serTag);
        if (!*ptr)
        {
            this->_errors->add(services::Error::create(services::ErrorObjectDoesNotSupportSerialization, services::SerializationTag, serTag));
            return;
        }

        (*ptr)->deserializeImpl(this);

        segmentFooter();
    }

    /**
     *  Performs data serialization creating a data segment
     *  \param[in]   obj  The serializable object
     */
    template <typename T>
    void setSharedPtrObj(services::SharedPtr<T> & obj) const
    {
        data_management::SerializationIface * ptr;
        setSingleObj(&ptr);
        if (this->_errors->size() != 0)
        {
            return;
        }
        if (ptr)
        {
            obj = services::SharedPtr<T>(static_cast<T *>(ptr));
        }
        else
        {
            obj = services::SharedPtr<T>();
        }
    }

    /**
     *  Performs deserialization of the objects stored in output data archive
     *  \return Shared pointer for the deserialized object
     */
    services::SharedPtr<SerializationIface> getAsSharedPtr() const
    {
        const int serTag = segmentHeader();
        services::SharedPtr<SerializationIface> ptr(Factory::instance().createObject(serTag));
        if (!ptr)
        {
            this->_errors->add(services::Error::create(services::ErrorObjectDoesNotSupportSerialization, services::SerializationTag, serTag));
            return services::SharedPtr<SerializationIface>();
        }
        ptr->deserializeImpl(this);
        segmentFooter();
        return ptr;
    }

    /**
     *  Returns the major version of the library used for object serialization
     *  \return Version of the library
     */
    int getMajorVersion() const { return _arch->getMajorVersion(); }

    /**
     *  Returns the minor version of the library used for object serialization
     *  \return Version of the library
     */
    int getMinorVersion() const { return _arch->getMinorVersion(); }

    /**
     *  Returns the update version of the library used for object serialization
     *  \return Version of the library
     */
    int getUpdateVersion() const { return _arch->getUpdateVersion(); }

    /**
    * Returns errors during the computation
    * \return Errors during the computation
    */
    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        if (_arch)
        {
            services::SharedPtr<services::ErrorCollection> errors = static_cast<DataArchiveImpl *>(_arch)->getErrors();
            if (errors.get())
            {
                _errors->add(*errors);
            }
        }
        return _errors;
    }

protected:
    DataArchiveIface * _arch;
    services::SharedPtr<services::ErrorCollection> _errors;

private:
    OutputDataArchive(const OutputDataArchive &);
    OutputDataArchive & operator=(const OutputDataArchive &);
};
/** @} */

} // namespace interface1
using interface1::DataArchiveIface;
using interface1::DataArchive;
using interface1::InputDataArchive;
using interface1::OutputDataArchive;

} // namespace data_management
} // namespace daal

#endif
