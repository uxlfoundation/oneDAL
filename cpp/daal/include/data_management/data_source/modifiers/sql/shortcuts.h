/* file: shortcuts.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_SQL_SHORTCUTS_H__
#define __DATA_SOURCE_MODIFIERS_SQL_SHORTCUTS_H__

#include "services/internal/error_handling_helpers.h"
#include "data_management/data_source/modifiers/sql/internal/default_modifiers.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
namespace sql
{
namespace interface1
{
/**
 * @ingroup data_source_modifiers_sql
 * @{
 */

/**
 * Simplifies creation of custom feature modifier
 * \tparam  Modifier  Type of modifier to be constructed
 * \return  Shared pointer to the modifier
 */
template <typename Modifier>
DAAL_DEPRECATED inline FeatureModifierIfacePtr custom()
{
    return services::internal::wrapSharedAndTryThrow<Modifier>(new Modifier());
}

/**
 * Creates continuous feature modifier which parses input column values as real numbers
 * \return Shared pointer to the continuous feature modifier
 */
DAAL_DEPRECATED inline FeatureModifierIfacePtr continuous()
{
    return services::internal::wrapSharedAndTryThrow<FeatureModifier>(new internal::ContinuousFeatureModifier());
}

/** @} */
} // namespace interface1

using interface1::custom;
using interface1::continuous;

} // namespace sql
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
