/* file: data_serialize.h */
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
//  Declaration and implementation of the serialization class
//--
*/

#ifndef __DAAL_SERIALIZE_H__
#define __DAAL_SERIALIZE_H__

#include "services/base.h"
#include "services/daal_memory.h"
#include "services/error_handling.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @defgroup serialization Data Serialization and Deserialization
 * \brief Contains classes that implement serialization and deserialization.
 * @ingroup data_management
 * @{
 */
class InputDataArchive;
class OutputDataArchive;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SERIALIZATIONIFACE"></a>
 *  \brief Abstract interface class that defines the interface for serialization and deserialization.
 */

class DAAL_EXPORT SerializationIface : public Base
{
public:
    virtual ~SerializationIface() override {}

    /**
     *  Performs serialization
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    void serialize(interface1::InputDataArchive & archive);

    /**
     *  Performs deserialization
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    void deserialize(interface1::OutputDataArchive & archive);

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    virtual int getSerializationTag() const = 0;

    /**
     *  Interfaces for the implementation of serialization
     *  \param[in]  archive  Storage for a serialized object or data structure
     */
    virtual services::Status serializeImpl(interface1::InputDataArchive * archive) = 0;

    /**
     *  Interfaces for the implementation of deserialization
     *  \param[in]  archive  Storage for a deserialized object or data structure
     */
    virtual services::Status deserializeImpl(const interface1::OutputDataArchive * archive) = 0;
};

/// @cond
/** For internal use only */
class DAAL_EXPORT SerializationDesc
{
public:
    typedef SerializationIface * (*creatorFunc)();
    SerializationDesc(creatorFunc func, int tag);
    int tag() const { return _tag; }
    creatorFunc creator() const { return _f; }
    const SerializationDesc * next() const { return _next; }
    static const SerializationDesc * first();

private:
    creatorFunc _f;
    const int _tag;
    const SerializationDesc * _next;
};
/// @endcond

/** @} */
} // namespace interface1
using interface1::SerializationIface;
using interface1::SerializationDesc;

} // namespace data_management
} // namespace daal
#define DECLARE_SERIALIZABLE_IMPL()                                                            \
    services::Status serializeImpl(data_management::InputDataArchive * arch) override          \
    {                                                                                          \
        return serialImpl<data_management::InputDataArchive, false>(arch);                     \
    }                                                                                          \
    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) override \
    {                                                                                          \
        return serialImpl<const data_management::OutputDataArchive, true>(arch);               \
    }

#define DECLARE_SERIALIZABLE()                       \
private:                                             \
    static data_management::SerializationDesc _desc; \
                                                     \
public:                                              \
    DECLARE_SERIALIZABLE_IMPL()                      \
    static int serializationTag();                   \
    virtual int getSerializationTag() const override;

#define DECLARE_SERIALIZABLE_IFACE()                 \
private:                                             \
    static data_management::SerializationDesc _desc; \
                                                     \
public:                                              \
    static int serializationTag();                   \
    virtual int getSerializationTag() const override;

#define DECLARE_SERIALIZABLE_TAG() \
public:                            \
    static int serializationTag(); \
    virtual int getSerializationTag() const override;

#define DECLARE_SERIALIZABLE_CAST(ClassName) \
    DECLARE_SERIALIZABLE()                   \
    DAAL_CAST_OPERATOR(ClassName)

#define DECLARE_MODEL(DstClassName, SrcClassName) \
    DECLARE_SERIALIZABLE()                        \
    DAAL_CAST_OPERATOR(DstClassName)              \
    DAAL_DOWN_CAST_OPERATOR(DstClassName, SrcClassName)

#define DECLARE_MODEL_IFACE(DstClassName, SrcClassName) \
    DECLARE_SERIALIZABLE_IFACE()                        \
    DAAL_CAST_OPERATOR(DstClassName)                    \
    DAAL_DOWN_CAST_OPERATOR(DstClassName, SrcClassName)

#endif
