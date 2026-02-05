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

#ifndef __DATA_SOURCE_MODIFIERS_SQL_INTERNAL_ENGINE_H__
#define __DATA_SOURCE_MODIFIERS_SQL_INTERNAL_ENGINE_H__

#include "services/internal/error_handling_helpers.h"
#include "data_management/features/internal/helpers.h"
#include "data_management/data_source/internal/sql_feature_utils.h"
#include "data_management/data_source/modifiers/internal/engine.h"
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
using data_management::internal::SQLFeatureInfo;
using data_management::internal::SQLFeaturesInfo;
using data_management::internal::SQLFetchBuffer;
using data_management::internal::SQLFetchBufferFragment;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__INTERNAL__FEATURECONFIG"></a>
 * \brief Class represents configuration of single input feature
 */
class InputFeatureInfo : public modifiers::internal::InputFeatureInfo
{
public:
    InputFeatureInfo() {}

    explicit InputFeatureInfo(const SQLFeatureInfo & featureInfo, const SQLFetchBufferFragment & fetchBuffer)
        : _featureInfo(featureInfo), _fetchBuffer(fetchBuffer)
    {}

    const SQLFeatureInfo & getFeatureInfo() const { return _featureInfo; }

    const SQLFetchBufferFragment & getFetchBuffer() const { return _fetchBuffer; }

private:
    SQLFeatureInfo _featureInfo;
    SQLFetchBufferFragment _fetchBuffer;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__INTERNAL__FEATURECONFIG"></a>
 * \brief Class represents configuration of single feature
 */
class OutputFeatureInfo : public modifiers::internal::OutputFeatureInfo
{};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__INTERNAL__CONFIGIMPL"></a>
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
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__INTERNAL__CONTEXTIMPL"></a>
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

    virtual size_t getNumberOfColumns() const override { return impl::getNumberOfInputFeatures(); }

    virtual services::BufferView<DAAL_DATA_TYPE> getOutputBuffer() const override { return impl::getOutputBuffer(); }

    virtual services::BufferView<char> getRawValue(size_t columnIndex) const override
    {
        const InputFeatureInfo & fi = impl::getPickedInputFeature(columnIndex);
        return fi.getFetchBuffer().view();
    }
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__SQL__INTERNAL__MODIFIERSMANAGER"></a>
 * \brief Class that holds modifiers and implements logic of modifiers applying flow
 */
class ModifiersManager : public modifiers::internal::ModifiersManager<FeatureModifierIface, ConfigImpl, ContextImpl>
{
public:
    static services::SharedPtr<ModifiersManager> create(services::Status * status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow(new ModifiersManager(), status);
    }

    services::Status prepare(const SQLFeaturesInfo & featuresInfo, const SQLFetchBuffer & fetchBuffer)
    {
        services::Status status;

        const features::FeatureIdMappingIfacePtr mapping = createMapping(featuresInfo, &status);
        DAAL_CHECK_STATUS_VAR(status);

        const services::internal::CollectionPtr<InputFeatureInfo> inputFeaturesInfo = createInputFeaturesInfo(featuresInfo, fetchBuffer, &status);
        DAAL_CHECK_STATUS_VAR(status);

        status |= getBinder().bind(mapping, inputFeaturesInfo);

        return status;
    }

private:
    ModifiersManager() {}

    services::internal::CollectionPtr<InputFeatureInfo> createInputFeaturesInfo(const SQLFeaturesInfo & featuresInfo,
                                                                                const SQLFetchBuffer & fetchBuffer, services::Status * status = NULL)
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
            const SQLFeatureInfo & fi         = featuresInfo[i];
            const SQLFetchBufferFragment & ff = fetchBuffer.getFragment(i);
            (*inputFeaturesInfo)[i]           = InputFeatureInfo(fi, ff);
        }

        return inputFeaturesInfo;
    }

    features::FeatureIdMappingIfacePtr createMapping(const SQLFeaturesInfo & featuresInfo, services::Status * status = NULL)
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

        for (size_t i = 0; i < numberOfFeatures; ++i)
        {
            mapping->setFeatureKey(i, featuresInfo[i].columnName);
        }

        return mapping;
    }
};
typedef services::SharedPtr<ModifiersManager> ModifiersManagerPtr;

} // namespace internal
} // namespace sql
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
