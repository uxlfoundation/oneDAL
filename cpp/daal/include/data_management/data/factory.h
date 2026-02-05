/* file: factory.h */
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
//  Implementation of service features used by the library components.
//--
*/

#ifndef __FACTORY_H__
#define __FACTORY_H__

#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_collection.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__ABSTRACTCREATOR"></a>
 *  \brief Interface class used by the Factory class to register and create objects of a specific class
 */
class DAAL_EXPORT AbstractCreator
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor */
    AbstractCreator() {}

    /** \private */
    virtual ~AbstractCreator() {}

    /**
     *  Creates a new object of a class
     *  \return Pointer to the new object
     */
    virtual SerializationIface * create() const = 0;

    /**
     *  Returns a unique class identifier associated with a class
     *  \return Class identifier
     */
    virtual int getTag() const = 0;
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CREATOR"></a>
 *  \brief Main class used by the Factory class to register and create objects of a class derived from SerializationIface
 *  and the default constructor without arguments
 *  \tparam  Derived  Object of this class is created by the create() function
 */
template <class Derived>
class Creator : public AbstractCreator
{
public:
    /** Default constructor */
    Creator() {}

    /** \private */
    virtual ~Creator() {}

    SerializationIface * create() const override { return new Derived(); }

    int getTag() const override { return Derived::serializationTag(); }
};

class FactoryImpl;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__FACTORY"></a>
 *  \brief Class that provides factory functionality for objects implementing the SerializationIface interface.
 *  Used within deserialization functionality.
 */
class DAAL_EXPORT Factory
{
public:
    /**
     *  Static function that returns an instance of the Factory class
     *  \return Reference to the Factory object
     */
    static Factory & instance();

    /**
     *  Registers the %Creator object for an additional class
     *  \param[in]  creator  Object that implements the AbstractCreator interface to create an instance of a class
     */
    void registerObject(AbstractCreator * creator);

    /**
     *  Creates a new object of a class described by an identifier
     *  \param[in]  objectId  Identifier of the class
     */
    SerializationIface * createObject(int objectId);

private:
    Factory();
    Factory(const Factory &);
    Factory & operator=(const Factory &);
    ~Factory();
    FactoryImpl * _impl;
};
/** @} */
} // namespace interface1
using interface1::AbstractCreator;
using interface1::Creator;
using interface1::Factory;

} // namespace data_management
} // namespace daal
#endif
