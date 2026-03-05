/* file: default_modifiers.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_SQL_DEFAULT_MODIFIERS_H__
#define __DATA_SOURCE_MODIFIERS_SQL_DEFAULT_MODIFIERS_H__

#include "services/daal_shared_ptr.h"
#include "services/internal/collection.h"

#include "data_management/features/defines.h"
#include "data_management/data_source/modifiers/sql/modifier.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
namespace sql
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__INTERNAL__CONTINUOUSFEATUREMODIFIER"></a>
 * \brief Feature modifier that parses tokens as continuous features
 */
class ContinuousFeatureModifier : public FeatureModifier
{
public:
    virtual void initialize(Config & config) override
    {
        const size_t numberOfFeatures = config.getNumberOfInputFeatures();
        for (size_t i = 0; i < numberOfFeatures; i++)
        {
            config.setOutputFeatureType(i, features::DAAL_CONTINUOUS);
        }
    }

    virtual void apply(Context & context) override
    {
        services::BufferView<DAAL_DATA_TYPE> outputBuffer = context.getOutputBuffer();
        for (size_t i = 0; i < outputBuffer.size(); i++)
        {
            outputBuffer[i] = context.getValue<DAAL_DATA_TYPE>(i);
        }
    }
};

} // namespace internal
} // namespace sql
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
