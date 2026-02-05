/* file: engine.h */
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

#ifndef __DATA_SOURCE_MODIFIERS_CSV_INTERNAL_ENGINE_H__
#define __DATA_SOURCE_MODIFIERS_CSV_INTERNAL_ENGINE_H__

#include <string>

#include "services/internal/error_handling_helpers.h"
#include "data_management/features/internal/helpers.h"
#include "data_management/features/internal/identifiers_impl.h"
#include "data_management/data_source/modifiers/csv/modifier.h"
#include "data_management/data_source/modifiers/internal/engine.h"
#include "data_management/data_source/internal/csv_feature_utils.h"

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
using data_management::internal::CSVFeaturesInfo;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__FEATURECONFIG"></a>
 * \brief Class represents configuration of single input feature
 */
class InputFeatureInfo : public modifiers::internal::InputFeatureInfo
{
public:
    InputFeatureInfo() : _token(), _detectedFeatureType(features::DAAL_CATEGORICAL) {}

    explicit InputFeatureInfo(features::FeatureType detectedFeatureType) : _detectedFeatureType(detectedFeatureType) {}

    const services::StringView & getToken() const { return _token; }

    features::FeatureType getDetectedFeatureType() const { return _detectedFeatureType; }

    void setToken(const services::StringView & token) { _token = token; }

private:
    services::StringView _token;
    features::FeatureType _detectedFeatureType;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__FEATURECONFIG"></a>
 * \brief Class represents configuration of single output feature
 */
class OutputFeatureInfo : public modifiers::internal::OutputFeatureInfo
{};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__CONFIGIMPL"></a>
 * \brief Internal implementation of feature modifier configuration
 */
class ConfigImpl : public Config, public modifiers::internal::Config<InputFeatureInfo, OutputFeatureInfo>
{
private:
    typedef modifiers::internal::Config<InputFeatureInfo, OutputFeatureInfo> impl;

public:
    ConfigImpl() {}

    explicit ConfigImpl(const services::internal::CollectionPtr<InputFeatureInfo *> & pickedInputFeatures, services::Status * status = NULL)
        : impl(pickedInputFeatures, status)
    {}

    virtual size_t getNumberOfInputFeatures() const override { return impl::getNumberOfInputFeatures(); }

    virtual features::FeatureType getInputFeatureDetectedType(size_t index) const override
    {
        return impl::getPickedInputFeature(index).getDetectedFeatureType();
    }

    virtual services::Status setNumberOfOutputFeatures(size_t numberOfOutputFeatures) override
    {
        return impl::setNumberOfOutputFeatures(numberOfOutputFeatures);
    }

    virtual services::Status setOutputFeatureType(size_t outputFeatureIndex, features::FeatureType featureType) override
    {
        return impl::setOutputFeatureType(outputFeatureIndex, featureType);
    }

    virtual services::Status setNumberOfCategories(size_t outputFeatureIndex, size_t numberOfCategories) override
    {
        return impl::setNumberOfCategories(outputFeatureIndex, numberOfCategories);
    }

    virtual services::Status setCategoricalDictionary(size_t outputFeatureIndex, const CategoricalFeatureDictionaryPtr & dictionary) override
    {
        return impl::setCategoricalDictionary(outputFeatureIndex, dictionary);
    }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__CONTEXTIMPL"></a>
 * \brief Internal implementation of feature modifier context
 */
class ContextImpl : public Context, public modifiers::internal::Context<InputFeatureInfo, OutputFeatureInfo>
{
private:
    typedef modifiers::internal::Context<InputFeatureInfo, OutputFeatureInfo> impl;

public:
    ContextImpl() {}

    explicit ContextImpl(const services::internal::CollectionPtr<InputFeatureInfo *> & pickedInputFeatures, services::Status * status = NULL)
        : impl(pickedInputFeatures, status)
    {}

    virtual size_t getNumberOfTokens() const override { return impl::getNumberOfInputFeatures(); }

    virtual services::StringView getToken(size_t index) const override { return impl::getPickedInputFeature(index).getToken(); }

    virtual services::BufferView<DAAL_DATA_TYPE> getOutputBuffer() const override { return impl::getOutputBuffer(); }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__CSV__INTERNAL__MODIFIERSMANAGER"></a>
 * \brief Class that holds modifiers and implements logic of modifiers applying flow
 */
class ModifiersManager : public modifiers::internal::ModifiersManager<FeatureModifierIface, ConfigImpl, ContextImpl>
{
public:
    static services::SharedPtr<ModifiersManager> create(services::Status * status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow<ModifiersManager>(new ModifiersManager(), status);
    }

    void setToken(size_t tokenIndex, const services::StringView & token) { getBinder().getInputFeatureInfo(tokenIndex).setToken(token); }

    services::Status prepare(const CSVFeaturesInfo & featuresInfo)
    {
        services::Status status;

        const features::FeatureIdMappingIfacePtr featureMapping = createFeatureMapping(featuresInfo, &status);
        DAAL_CHECK_STATUS_VAR(status);

        const services::internal::CollectionPtr<InputFeatureInfo> inputFeaturesInfo = createInputFeaturesInfo(featuresInfo, &status);
        DAAL_CHECK_STATUS_VAR(status);

        status |= getBinder().bind(featureMapping, inputFeaturesInfo);

        return status;
    }

private:
    services::internal::CollectionPtr<InputFeatureInfo> createInputFeaturesInfo(const CSVFeaturesInfo & featuresInfo,
                                                                                services::Status * status = NULL)
    {
        const size_t numberOfFeatures = featuresInfo.getNumberOfFeatures();

        services::internal::CollectionPtr<InputFeatureInfo> inputFeaturesInfo =
            services::internal::HeapAllocatableCollection<InputFeatureInfo>::create(numberOfFeatures);
        if (!inputFeaturesInfo)
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorMemoryAllocationFailed);
            return inputFeaturesInfo;
        }

        for (size_t i = 0; i < numberOfFeatures; i++)
        {
            features::FeatureType fType = featuresInfo.getDetectedFeatureType(i);
            (*inputFeaturesInfo)[i]     = InputFeatureInfo(fType);
        }

        return inputFeaturesInfo;
    }

    features::FeatureIdMappingIfacePtr createFeatureMapping(const CSVFeaturesInfo & featuresInfo, services::Status * status = NULL)
    {
        const size_t numberOfFeatures = featuresInfo.getNumberOfFeatures();

        services::Status localStatus;
        const features::internal::FeatureIdDefaultMappingPtr mapping =
            features::internal::FeatureIdDefaultMapping::create(numberOfFeatures, &localStatus);
        if (!localStatus)
        {
            services::internal::tryAssignStatusAndThrow(status, localStatus);
            return mapping;
        }

        if (featuresInfo.areFeatureNamesAvailable())
        {
            for (size_t i = 0; i < numberOfFeatures; ++i)
            {
                mapping->setFeatureKey(i, featuresInfo.getFeatureName(i));
            }
        }

        return mapping;
    }

private:
    ModifiersManager() {}
};
typedef services::SharedPtr<ModifiersManager> ModifiersManagerPtr;

} // namespace internal
} // namespace csv
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
