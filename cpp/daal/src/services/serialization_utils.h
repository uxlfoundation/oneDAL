/* file: serialization_utils.h */
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
//  Declaration and implementation of mechanism for serializable objects registration.
//--
*/

#ifndef __SERIALIZATION_UTILS_H__
#define __SERIALIZATION_UTILS_H__

#include "data_management/data/factory.h"

#define __DAAL_SERIALIZATION_TAG(ClassName, Tag) \
    int ClassName::serializationTag()            \
    {                                            \
        return _desc.tag();                      \
    }                                            \
    int ClassName::getSerializationTag() const   \
    {                                            \
        return _desc.tag();                      \
    }

#define __DAAL_REGISTER_SERIALIZATION_CLASS(ClassName, Tag)                       \
    static data_management::SerializationIface * creator##ClassName()             \
    {                                                                             \
        return new ClassName();                                                   \
    }                                                                             \
    data_management::SerializationDesc ClassName::_desc(creator##ClassName, Tag); \
    __DAAL_SERIALIZATION_TAG(ClassName, Tag)

#define __DAAL_REGISTER_SERIALIZATION_CREATE_CLASS(ClassName, Tag)                \
    static data_management::SerializationIface * creator##ClassName()             \
    {                                                                             \
        return ClassName::create();                                               \
    }                                                                             \
    data_management::SerializationDesc ClassName::_desc(creator##ClassName, Tag); \
    __DAAL_SERIALIZATION_TAG(ClassName, Tag)

#define __DAAL_REGISTER_SERIALIZATION_CLASS2(ClassName, ImplClassName, Tag)       \
    static data_management::SerializationIface * creator##ClassName()             \
    {                                                                             \
        return new ImplClassName();                                               \
    }                                                                             \
    data_management::SerializationDesc ClassName::_desc(creator##ClassName, Tag); \
    __DAAL_SERIALIZATION_TAG(ClassName, Tag)

#define __DAAL_REGISTER_SERIALIZATION_CLASS3(ClassName, ClassName2, Tag)                                  \
    static data_management::SerializationIface * creator##ClassName##ClassName2()                         \
    {                                                                                                     \
        return new ClassName<ClassName2>();                                                               \
    }                                                                                                     \
    data_management::SerializationDesc ClassName<ClassName2>::_desc(creator##ClassName##ClassName2, Tag); \
    __DAAL_SERIALIZATION_TAG(ClassName<ClassName2>, Tag)

#endif
