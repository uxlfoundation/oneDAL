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

#ifndef __DATA_SOURCE_MODIFIERS_CSV_DEFAULT_MODIFIERS_H__
#define __DATA_SOURCE_MODIFIERS_CSV_DEFAULT_MODIFIERS_H__

#include "services/daal_shared_ptr.h"
#include "services/internal/collection.h"

#include "data_management/features/defines.h"
#include "data_management/data_source/modifiers/csv/modifier.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
namespace csv
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__FEATUREMODIFIERPRIMITIVE"></a>
 * \brief Primitive modifier that applicable to a single column
 */
class FeatureModifierPrimitive : public Base
{
public:
    virtual void initialize(Config & /*context*/, size_t /*index*/) {}
    virtual DAAL_DATA_TYPE apply(Context & context, size_t index) = 0;
    virtual void finalize(Config & /*context*/, size_t /*index*/) {}
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__DEFAULTFEATUREMODIFIERPRIMITIVE"></a>
 * \brief Default implementation of primitive feature modifier
 */
class DefaultFeatureModifierPrimitive : public FeatureModifierPrimitive
{
public:
    virtual DAAL_DATA_TYPE apply(Context & /*context*/, size_t /*index*/) override { return (DAAL_DATA_TYPE)0; }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__CONTINUOUSFEATUREMODIFIERPRIMITIVE"></a>
 * \brief Primitive feature modifier that parses tokens as continuous features
 */
class ContinuousFeatureModifierPrimitive : public FeatureModifierPrimitive
{
public:
    virtual void initialize(Config & config, size_t index) override { config.setOutputFeatureType(index, features::DAAL_CONTINUOUS); }

    virtual DAAL_DATA_TYPE apply(Context & context, size_t index) override { return context.getTokenAs<DAAL_DATA_TYPE>(index); }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__CATEGORICALFEATUREMODIFIERPRIMITIVE"></a>
 * \brief Primitive feature modifier that parses tokens as categorical features
 */
class CategoricalFeatureModifierPrimitive : public FeatureModifierPrimitive
{
public:
    CategoricalFeatureModifierPrimitive() : _catDict(new CategoricalFeatureDictionary()) {}

    virtual void initialize(Config & config, size_t index) override { config.setOutputFeatureType(index, features::DAAL_CATEGORICAL); }

    virtual DAAL_DATA_TYPE apply(Context & context, size_t index) override
    {
        const services::StringView token = context.getToken(index);
        const std::string sToken(token.begin(), token.end());
        const CategoricalFeatureDictionary::iterator it = _catDict->find(sToken);

        if (it != _catDict->end())
        {
            it->second.second++;
            return (DAAL_DATA_TYPE)it->second.first;
        }
        else
        {
            const int itemIndex = (int)(_catDict->size());
            const std::pair<int, int> indexPair(itemIndex, 1);
            (*_catDict)[sToken] = indexPair;
            return (DAAL_DATA_TYPE)itemIndex;
        }
    }

    virtual void finalize(Config & config, size_t index) override
    {
        const size_t numberOfCategories = _catDict->size();
        config.setNumberOfCategories(index, numberOfCategories);
        config.setCategoricalDictionary(index, _catDict);
    }

private:
    CategoricalFeatureDictionaryPtr _catDict;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__CONTINUOUSFEATUREMODIFIER"></a>
 * \brief Feature modifier that parses tokens as continuous features
 */
class ContinuousFeatureModifier : public FeatureModifier
{
public:
    virtual void initialize(Config & config) override
    {
        FeatureModifier::initialize(config);

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
            outputBuffer[i] = context.getTokenAs<DAAL_DATA_TYPE>(i);
        }
    }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__CATEGORICALFEATUREMODIFIER"></a>
 * \brief Feature modifier that parses tokens as categorical features
 */
class CategoricalFeatureModifier : public FeatureModifier
{
public:
    virtual void initialize(Config & config) override
    {
        FeatureModifier::initialize(config);

        const size_t numberOfInputFeatures = config.getNumberOfInputFeatures();
        _primitives                        = services::Collection<CategoricalFeatureModifierPrimitive>(numberOfInputFeatures);
        if (!_primitives.data())
        {
            services::throwIfPossible(services::ErrorMemoryAllocationFailed);
            return;
        }

        for (size_t i = 0; i < numberOfInputFeatures; i++)
        {
            _primitives[i].initialize(config, i);
        }
    }

    virtual void apply(Context & context) override
    {
        services::BufferView<DAAL_DATA_TYPE> outputBuffer = context.getOutputBuffer();
        for (size_t i = 0; i < outputBuffer.size(); i++)
        {
            outputBuffer[i] = _primitives[i].apply(context, i);
        }
    }

    virtual void finalize(Config & config) override
    {
        FeatureModifier::finalize(config);

        const size_t numberOfOutputFeatures = config.getNumberOfInputFeatures();
        for (size_t i = 0; i < numberOfOutputFeatures; i++)
        {
            _primitives[i].finalize(config, i);
        }
    }

private:
    services::Collection<CategoricalFeatureModifierPrimitive> _primitives;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__AUTOMATICFEATUREMODIFIER"></a>
 * \brief Feature modifier that determines suitable feature type and parses tokens according to determined type
 */
class AutomaticFeatureModifier : public FeatureModifier
{
public:
    virtual void initialize(Config & config) override
    {
        FeatureModifier::initialize(config);

        const size_t numberOfInputFeatures = config.getNumberOfInputFeatures();
        for (size_t i = 0; i < numberOfInputFeatures; i++)
        {
            FeatureModifierPrimitive * primitive = createPrimitive(config.getInputFeatureDetectedType(i));

            if (!_primitives.push_back(primitive))
            {
                services::throwIfPossible(services::ErrorMemoryAllocationFailed);
                return;
            }

            primitive->initialize(config, i);
        }
    }

    virtual void apply(Context & context) override
    {
        services::BufferView<DAAL_DATA_TYPE> outputBuffer = context.getOutputBuffer();
        for (size_t i = 0; i < outputBuffer.size(); i++)
        {
            outputBuffer[i] = _primitives[i].apply(context, i);
        }
    }

    virtual void finalize(Config & config) override
    {
        FeatureModifier::finalize(config);

        const size_t numberOfOutputFeatures = config.getNumberOfInputFeatures();
        for (size_t i = 0; i < numberOfOutputFeatures; i++)
        {
            _primitives[i].finalize(config, i);
        }
    }

private:
    FeatureModifierPrimitive * createPrimitive(features::FeatureType featureType)
    {
        switch (featureType)
        {
        case features::DAAL_CONTINUOUS: return new ContinuousFeatureModifierPrimitive();

        case features::DAAL_ORDINAL:
        case features::DAAL_CATEGORICAL: return new CategoricalFeatureModifierPrimitive();
        }
        return new DefaultFeatureModifierPrimitive();
    }

private:
    services::internal::ObjectPtrCollection<FeatureModifierPrimitive> _primitives;
};
typedef services::SharedPtr<AutomaticFeatureModifier> AutomaticFeatureModifierPtr;

} // namespace internal
} // namespace csv
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
