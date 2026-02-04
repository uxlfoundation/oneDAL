/* file: data_collection.h */
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

#ifndef __DATA_COLLECTION_H__
#define __DATA_COLLECTION_H__

#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_archive.h"
#include "services/daal_shared_ptr.h"
#include "services/collection.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @defgroup data_model Data Model
 * \brief Contains classes that provide functionality of Collection container for objects derived from SerializationIface interface and implements SerializationIface itself
 * @ingroup data_management
 * @{
 */
typedef services::SharedPtr<SerializationIface> SerializationIfacePtr;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATACOLLECTION"></a>
 *  \brief Class that provides functionality of Collection container for objects derived from
 *  SerializationIface interface and implements SerializationIface itself
 */
class DAAL_EXPORT DataCollection : public SerializationIface, private services::Collection<SerializationIfacePtr>
{
public:
    DECLARE_SERIALIZABLE_TAG()

    typedef services::Collection<SerializationIfacePtr> super;

    DAAL_CAST_OPERATOR(DataCollection)

    /** Default constructor */
    DataCollection();

    /** Copy constructor */
    DataCollection(const DataCollection & other);

    /**
     *  Constructor with a defined number of elements
     *  \param[in]  n  Number of elements
     */
    DataCollection(size_t n);

    virtual ~DataCollection() {};

    /**
    *  Const element access
    *  \param[in] index Index of an accessed element
    *  \return    Pointer to the element
    */
    const SerializationIfacePtr & operator[](size_t index) const;

    /**
    *  Element access
    *  \param[in] index Index of an accessed element
    *  \return    Pointer to the element
    */
    SerializationIfacePtr & operator[](size_t index);

    /**
    *  Element access
    *  \param[in] index Index of an accessed element
    *  \return    Reference to the element
    */
    SerializationIfacePtr & get(size_t index);

    /**
    *  Const element access
    *  \param[in] index Index of an accessed element
    *  \return    Reference to the element
    */
    const SerializationIfacePtr & get(size_t index) const;

    /**
    *  Adds an element to the end of a collection
    *  \param[in] x Element to add
    */
    DataCollection & push_back(const SerializationIfacePtr & x);

    /**
     *  Adds an element to the end of a collection
     *  \param[in] x Element to add
     */
    DataCollection & operator<<(const SerializationIfacePtr & x);

    /**
     *  Size of a collection
     *  \return Size of the collection
     */
    size_t size() const;

    /**
    *  Clears a collection: removes an array, sets the size and capacity to 0
    */
    void clear();

    /**
     *  Erase an element from a position
     *  \param[in] pos Position to erase
     */
    void erase(size_t pos);

    /**
    *  Changes the size of a storage
    *  \param[in] newCapacity Size of a new storage.
    */
    bool resize(size_t newCapacity);

    services::Status serializeImpl(interface1::InputDataArchive * arch) override { return serialImpl<interface1::InputDataArchive, false>(arch); }

    services::Status deserializeImpl(const interface1::OutputDataArchive * arch) override
    {
        return serialImpl<const interface1::OutputDataArchive, true>(arch);
    }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        size_t size = _size;

        arch->set(size);

        if (onDeserialize)
        {
            resize(size);
        }

        _size = size;

        for (size_t i = 0; i < _size; i++)
        {
            arch->setSharedPtrObj(_array[i]);
        }

        return services::Status();
    }
};
typedef services::SharedPtr<DataCollection> DataCollectionPtr;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__KEYVALUECOLLECTION"></a>
 *  \brief Class that provides functionality of a key-value container for objects derived from the
 *  T with a key of the size_t type
 */
template <typename T>
class DAAL_EXPORT KeyValueCollection
{
public:
    DAAL_CAST_OPERATOR(KeyValueCollection)

    /** Default constructor */
    KeyValueCollection() {}
    /** Copy constructor */
    KeyValueCollection(const KeyValueCollection & other) : _keys(other._keys), _values(other._values) {}

    KeyValueCollection(const services::Collection<size_t> & keys, const services::Collection<services::SharedPtr<T> > & values)
    {
        for (size_t i = 0; i < keys.size(); i++)
        {
            _keys.push_back(keys[i]);
            _values.push_back(values[i]);
        }
    }

    virtual ~KeyValueCollection() {}

    /**
     *  Returns a reference to SharedPtr for a stored object with a given key if an object with such key was registered
     *  \param[in]  k  Key value
     *  \return Reference to SharedPtr of the SerializationIface type
     */
    const services::SharedPtr<T> & operator[](size_t k) const
    {
        size_t i;
        for (i = 0; i < _keys.size(); i++)
        {
            if (_keys[i] == k)
            {
                return _values[i];
            }
        }
        return _nullPtr;
    }

    /**
     *  Creates an empty SharedPtr and stores it under a requested key and returns a reference for this value
     *  \param[in]  k  Key value
     *  \return Reference to SharedPtr of the SerializationIface type
     */
    services::SharedPtr<T> & operator[](size_t k);

    /**
     *  Returns a reference to SharedPtr for a stored key with a given index
     *  \param[in]  idx  Index of the requested key
     *  \return Reference to SharedPtr of the size_t type
     */
    size_t getKeyByIndex(int idx) const { return _keys[idx]; }

    /**
     *  Returns a reference to SharedPtr for a stored object with a given index
     *  \param[in]  idx  Index of the requested object
     *  \return Reference to SharedPtr of the SerializationIface type
     */
    services::SharedPtr<T> & getValueByIndex(int idx) { return _values[idx]; }

    /**
     *  Returns a const SharedPtr for a stored object with a given index
     *  \param[in]  idx  Index of the requested object
     *  \return Reference to SharedPtr of the SerializationIface type
     */
    const services::SharedPtr<T> getValueByIndex(int idx) const { return _values[idx]; }

    /**
     *  Returns the number of stored objects
     *  \return Number of stored objects
     */
    size_t size() const { return _keys.size(); }

    /**
     *  Removes all elements from a container
     */
    void clear()
    {
        _keys.clear();
        _values.clear();
    }

protected:
    services::Collection<size_t> _keys;
    services::Collection<services::SharedPtr<T> > _values;
    services::SharedPtr<T> _nullPtr;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SerializableKeyValueCollection"></a>
 *  \brief Class that provides functionality of a key-value container for objects derived from the
 *  SerializationIface interface with a key of the size_t type
 */
template <typename T>
class DAAL_EXPORT SerializableKeyValueCollection : public SerializationIface, public KeyValueCollection<T>
{
public:
    DECLARE_SERIALIZABLE_TAG()

    DAAL_CAST_OPERATOR(SerializableKeyValueCollection)

    /** Default constructor */
    SerializableKeyValueCollection() : KeyValueCollection<T>() {}
    /** Copy constructor */
    SerializableKeyValueCollection(const SerializableKeyValueCollection & other) : KeyValueCollection<T>(other) {}

    SerializableKeyValueCollection(const services::Collection<size_t> & keys, const services::Collection<services::SharedPtr<T> > & values)
        : KeyValueCollection<T>(keys, values)
    {}

    virtual ~SerializableKeyValueCollection() {}

    /** \private */
    services::Status serializeImpl(interface1::InputDataArchive * arch) override { return serialImpl<interface1::InputDataArchive, false>(arch); }

    /** \private */
    services::Status deserializeImpl(const interface1::OutputDataArchive * arch) override
    {
        return serialImpl<const interface1::OutputDataArchive, true>(arch);
    }

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        size_t size = this->_values.size();

        arch->set(size);

        if (onDeserialize)
        {
            this->_values.resize(size);
            this->_keys.resize(size);
        }

        for (size_t i = 0; i < size; i++)
        {
            if (onDeserialize)
            {
                this->_values.push_back(this->_nullPtr);
                this->_keys.push_back(0);
            }
            arch->setSharedPtrObj(this->_values[i]);
            arch->set(this->_keys[i]);
        }

        return services::Status();
    }
};

typedef SerializableKeyValueCollection<SerializationIface> KeyValueDataCollection;
typedef services::SharedPtr<KeyValueDataCollection> KeyValueDataCollectionPtr;
typedef services::SharedPtr<const KeyValueDataCollection> KeyValueDataCollectionConstPtr;
/** @} */
} // namespace interface1
using interface1::DataCollection;
using interface1::DataCollectionPtr;
using interface1::KeyValueCollection;
using interface1::SerializableKeyValueCollection;
using interface1::KeyValueDataCollection;
using interface1::KeyValueDataCollectionPtr;
using interface1::KeyValueDataCollectionConstPtr;
using interface1::SerializationIfacePtr;

} // namespace data_management
} // namespace daal

#endif
