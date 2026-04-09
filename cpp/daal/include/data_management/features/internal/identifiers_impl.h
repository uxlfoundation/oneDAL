/* file: identifiers_impl.h */
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

#ifndef __DATA_MANAGEMENT_FEATURES_INTERNAL_IDENTIFIERS_IMPL_H__
#define __DATA_MANAGEMENT_FEATURES_INTERNAL_IDENTIFIERS_IMPL_H__

#include <map>
#include <limits>
#include <string>

#include "services/collection.h"
#include "services/internal/utilities.h"
#include "data_management/features/identifiers.h"
#include "data_management/features/internal/indices_impl.h"

namespace daal
{
namespace data_management
{
namespace features
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__FEATUREIDDEFAULTMAPPING"></a>
 * \brief Default implementation of feature mapping
 */
class FeatureIdDefaultMapping : public FeatureIdMapping
{
private:
    typedef std::map<std::string, FeatureIndex> KeyToIndexMap;

public:
    static services::SharedPtr<FeatureIdDefaultMapping> create(size_t numberOfFeatures, services::Status * status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow<FeatureIdDefaultMapping>(new FeatureIdDefaultMapping(numberOfFeatures), status);
    }

    size_t getNumberOfFeatures() const override { return _numberOfFeatures; }

    FeatureIndex getIndexByKey(const services::String & key) const override
    {
        const std::string stdKey(key.c_str(), key.c_str() + key.length());
        KeyToIndexMap & keyToIndexMap    = const_cast<KeyToIndexMap &>(_keyToIndexMap);
        KeyToIndexMap::const_iterator it = keyToIndexMap.find(stdKey);
        if (it == keyToIndexMap.end())
        {
            return FeatureIndexTraits::invalid();
        }
        return it->second;
    }

    bool areKeysAvailable() const override { return _keyToIndexMap.size() > 0; }

    void setFeatureKey(FeatureIndex featureIndex, const services::String & key)
    {
        const std::string stdKey(key.c_str(), key.c_str() + key.length());
        _keyToIndexMap[stdKey] = featureIndex;
    }

    void setNumberOfFeatures(size_t numberOfFeatures) { _numberOfFeatures = numberOfFeatures; }

private:
    explicit FeatureIdDefaultMapping(size_t numberOfFeatures) : _numberOfFeatures(numberOfFeatures), _keyToIndexMap() {}

    size_t _numberOfFeatures;
    KeyToIndexMap _keyToIndexMap;
};
typedef services::SharedPtr<FeatureIdDefaultMapping> FeatureIdDefaultMappingPtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__FEATUREIDLIST"></a>
 * \brief Implementation of FeatureIdCollection to store a list of feature identifiers
 */
class FeatureIdList : public FeatureIdCollection
{
public:
    static services::SharedPtr<FeatureIdList> create(services::Status * status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow<FeatureIdList>(new FeatureIdList(), status);
    }

    FeatureIndicesIfacePtr mapToFeatureIndices(const FeatureIdMappingIface & mapping, services::Status * status = NULL) override
    {
        const size_t numberOfFeatures = _featureIds.size();

        services::Status localStatus;

        FeatureIndicesListPtr featureFeatureIndices = FeatureIndicesList::create(&localStatus);
        services::internal::tryAssignStatusAndThrow(status, localStatus);
        if (!localStatus.ok())
        {
            return FeatureIndicesPtr();
        }

        for (size_t i = 0; i < numberOfFeatures; i++)
        {
            const FeatureIdIfacePtr & id = _featureIds[i];

            const FeatureIndex mappedIndex = id->mapToIndex(mapping, &localStatus);
            services::internal::tryAssignStatusAndThrow(status, localStatus);
            if (!localStatus.ok())
            {
                return FeatureIndicesPtr();
            }

            featureFeatureIndices->add(mappedIndex);
        }

        return featureFeatureIndices;
    }

    services::Status add(const FeatureIdIfacePtr & id)
    {
        if (!id)
        {
            return services::throwIfPossible(services::ErrorNullPtr);
        }

        if (!_featureIds.safe_push_back(id))
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        return services::Status();
    }

    size_t size() const { return _featureIds.size(); }

private:
    FeatureIdList() {}

    services::Collection<FeatureIdIfacePtr> _featureIds;
};
typedef services::SharedPtr<FeatureIdList> FeatureIdListPtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__FEATUREIDRANGE"></a>
 * \brief Implementation of FeatureIdCollection to store a range of feature identifiers
 */
class FeatureIdRange : public FeatureIdCollection
{
public:
    static services::SharedPtr<FeatureIdRange> create(const FeatureIdIfacePtr & begin, const FeatureIdIfacePtr & end,
                                                      services::Status * status = NULL)
    {
        if (!begin || !end)
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorNullPtr);
            return services::SharedPtr<FeatureIdRange>();
        }

        return services::internal::wrapSharedAndTryThrow<FeatureIdRange>(new FeatureIdRange(begin, end), status);
    }

    FeatureIndicesIfacePtr mapToFeatureIndices(const FeatureIdMappingIface & mapping, services::Status * status = NULL) override
    {
        services::Status localStatus;

        const FeatureIndex beginIndex = _begin->mapToIndex(mapping, &localStatus);
        services::internal::tryAssignStatusAndThrow(status, localStatus);

        const FeatureIndex endIndex = _end->mapToIndex(mapping, &localStatus);
        services::internal::tryAssignStatusAndThrow(status, localStatus);

        if (!localStatus.ok())
        {
            return FeatureIndicesPtr();
        }
        return FeatureIndicesRange::create(beginIndex, endIndex, status);
    }

private:
    explicit FeatureIdRange(const FeatureIdIfacePtr & begin, const FeatureIdIfacePtr & end) : _begin(begin), _end(end) {}

    FeatureIdIfacePtr _begin;
    FeatureIdIfacePtr _end;
};
typedef services::SharedPtr<FeatureIdRange> FeatureIdRangePtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__NUMERICFEATUREID"></a>
 * \brief Implementation of FeatureId that uses number as a reference to particular feature
 */
class NumericFeatureId : public FeatureId
{
public:
    typedef long long InternalIndex;

    static services::SharedPtr<NumericFeatureId> create(NumericFeatureId::InternalIndex index, services::Status * status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow<NumericFeatureId>(new NumericFeatureId(index), status);
    }

    FeatureIndex mapToIndex(const FeatureIdMappingIface & mapping, services::Status * status = NULL) override
    {
        const size_t numberOfFeatures = mapping.getNumberOfFeatures();

        /* Check that size of 'InternalIndex' type is enough to store number of
         * features and is able to store intermediate result of computations */
        DAAL_OVERFLOW_CHECK_BY_ADDING(InternalIndex, numberOfFeatures, _index);

        /* Positive indices in the range [0, nf - 1] and negative indices in the range [ -nf, -1 ] are allowed.
         * Negative indices should be interpreted as a result of subtraction nf - abs(index) */
        const InternalIndex nf = (InternalIndex)numberOfFeatures;
        if (_index >= nf || _index < -nf)
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorIncorrectIndex);
            return FeatureIndexTraits::invalid();
        }

        return (FeatureIndex)((nf + _index) % nf);
    }

private:
    explicit NumericFeatureId(InternalIndex index) : _index(index) {}

    InternalIndex _index;
};
typedef services::SharedPtr<NumericFeatureId> NumericFeatureIdPtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__STRINGFEATUREID"></a>
 * \brief Implementation of FeatureId that uses string as a reference to particular feature
 */
class StringFeatureId : public FeatureId
{
public:
    static services::SharedPtr<StringFeatureId> create(const services::String & name, services::Status * status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow<StringFeatureId>(new StringFeatureId(name), status);
    }

    FeatureIndex mapToIndex(const FeatureIdMappingIface & mapping, services::Status * status = NULL) override
    {
        if (!mapping.areKeysAvailable())
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorFeatureNamesNotAvailable);
            return FeatureIndexTraits::invalid();
        }

        const FeatureIndex index = mapping.getIndexByKey(_name);
        if (index == FeatureIndexTraits::invalid())
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorFeatureNamesNotAvailable);
            return FeatureIndexTraits::invalid();
        }

        return index;
    }

private:
    explicit StringFeatureId(const services::String & name) : _name(name) {}

    services::String _name;
};
typedef services::SharedPtr<StringFeatureId> StringFeatureIdPtr;

} // namespace internal
} // namespace features
} // namespace data_management
} // namespace daal

#endif
