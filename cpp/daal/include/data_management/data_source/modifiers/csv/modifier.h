/* file: modifier.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_CSV_MODIFIER_H__
#define __DATA_SOURCE_MODIFIERS_CSV_MODIFIER_H__

#include "services/daal_string.h"
#include "data_management/data_source/modifiers/modifier.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
/**
 * \brief Contains modifiers components for CSV Data Source
 */
namespace csv
{
/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @defgroup data_source_modifiers_csv CSV
 * \brief Defines CSV specific feature modifiers
 * @ingroup data_source_modifiers
 * @{
 */

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__CONFIGIFACE"></a>
 * \brief Abstract class that defines interface of modifier configuration
 */
class ConfigIface : public modifiers::ConfigIface
{
public:
    /**
     * Gets automatically detected type of the input feature
     * \param[in] inputFeatureIndex  The input feature index
     * \return    The type of input feature type
     */
    virtual features::FeatureType getInputFeatureDetectedType(size_t inputFeatureIndex) const = 0;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__CONFIG"></a>
 * \brief Base class that represents modifier configuration, object of that
 *        class is passed to the modifier on initialization and finalization stages
 */
class Config : public Base, public ConfigIface
{};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__CONTEXTIFACE"></a>
 * \brief Abstract class that defines interface of modifier context
 */
class ContextIface : public modifiers::ContextIface
{
public:
    /**
     * Gets the number of tokens available for the modifier.
     * \return The number of tokens
     */
    virtual size_t getNumberOfTokens() const = 0;

    /**
     * Gets the token. One token corresponds to one delimiter-separated word in the row of CSV file
     * \param[in]  index  The index of token
     * \return     The corresponding token
     */
    virtual services::StringView getToken(size_t index) const = 0;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__CONTEXT"></a>
 * \brief Base class that represents modifier context, object of that class is
 *        passed to the modifier as an argument of FeatureModifierIface::apply method
 */
class Context : public Base, public ContextIface
{
public:
    /**
     * Gets the parsed token and tries to cast it to specified type
     * \tparam     T Type to which token should be casted
     * \param[in]  index  The index of token
     * \return     The parsed token
     */
    template <typename T>
    T getTokenAs(size_t index) const;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__FEATUREMODIFIERIFACE"></a>
 * \brief Specialization of modifiers::FeatureModifierIface for CSV feature modifier
 */
class FeatureModifierIface : public modifiers::FeatureModifierIface<Config, Context>
{};
typedef services::SharedPtr<FeatureModifierIface> FeatureModifierIfacePtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__FEATUREMODIFIER"></a>
 * \brief Base class for feature modifier, intended for inheritance from the user side
 */
class FeatureModifier : public Base, public FeatureModifierIface
{
public:
    /**
     * Default implementation of interface method
     * \param config  The configuration of the modifier
     */
    virtual void initialize(Config & /*config*/) DAAL_C11_OVERRIDE {}

    /**
     * Default implementation of interface method
     * \param config  The configuration of the modifier
     */
    virtual void finalize(Config & /*config*/) DAAL_C11_OVERRIDE {}
};
typedef services::SharedPtr<FeatureModifier> FeatureModifierPtr;

/* Specifications of the Context::getTokenAs method */

template <>
inline float Context::getTokenAs<float>(size_t index) const
{
    return (float)services::daal_string_to_float(getToken(index).c_str(), 0);
}

template <>
inline double Context::getTokenAs<double>(size_t index) const
{
    return (double)services::daal_string_to_double(getToken(index).c_str(), 0);
}

template <>
inline services::StringView Context::getTokenAs<services::StringView>(size_t index) const
{
    return getToken(index);
}

template <>
inline std::string Context::getTokenAs<std::string>(size_t index) const
{
    services::StringView token = getToken(index);
    return std::string(token.begin(), token.end());
}

/** @} */
} // namespace interface1

using interface1::Config;
using interface1::Context;
using interface1::FeatureModifierIface;
using interface1::FeatureModifierIfacePtr;
using interface1::FeatureModifier;
using interface1::FeatureModifierPtr;

} // namespace csv
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
