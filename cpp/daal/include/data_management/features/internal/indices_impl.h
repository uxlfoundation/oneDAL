/* file: indices_impl.h */
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

#ifndef __DATA_MANAGEMENT_FEATURES_INTERNAL_INDICES_IMPL_H__
#define __DATA_MANAGEMENT_FEATURES_INTERNAL_INDICES_IMPL_H__

#include <map>
#include <string>

#include "services/collection.h"
#include "services/internal/utilities.h"
#include "services/internal/error_handling_helpers.h"

#include "data_management/features/indices.h"

namespace daal
{
namespace data_management
{
namespace features
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__FEATUREINDICESLIST"></a>
 * \brief Implementation of FeatureIndices to store a list of feature indices
 */
class FeatureIndicesList : public FeatureIndices
{
public:
    static services::SharedPtr<FeatureIndicesList> create(services::Status * status = NULL)
    {
        return services::internal::wrapSharedAndTryThrow<FeatureIndicesList>(new FeatureIndicesList(), status);
    }

    virtual size_t size() const override { return _indices.size(); }

    virtual bool isPlainRange() const override { return false; }

    virtual bool areRawFeatureIndicesAvailable() const override { return true; }

    virtual FeatureIndex getFirst() const override
    {
        if (!size())
        {
            return FeatureIndexTraits::invalid();
        }
        return _indices[0];
    }

    virtual FeatureIndex getLast() const override
    {
        if (!size())
        {
            return FeatureIndexTraits::invalid();
        }
        return _indices[_indices.size() - 1];
    }

    virtual services::BufferView<FeatureIndex> getRawFeatureIndices() override
    {
        return services::BufferView<FeatureIndex>(_indices.data(), _indices.size());
    }

    services::Status add(FeatureIndex index)
    {
        if (index > FeatureIndexTraits::maxIndex() || index == FeatureIndexTraits::invalid())
        {
            return services::throwIfPossible(services::ErrorIncorrectDataRange);
        }

        if (!_indices.safe_push_back(index))
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        return services::Status();
    }

private:
    FeatureIndicesList() {}

    services::Collection<FeatureIndex> _indices;
};
typedef services::SharedPtr<FeatureIndicesList> FeatureIndicesListPtr;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__FEATUREINDICESRANGE"></a>
 * \brief Implementation of FeatureIndices to store a range of feature indices
 */
class FeatureIndicesRange : public FeatureIndices
{
public:
    static services::SharedPtr<FeatureIndicesRange> create(FeatureIndex begin, FeatureIndex end, services::Status * status = NULL)
    {
        if (begin == FeatureIndexTraits::invalid() || end == FeatureIndexTraits::invalid())
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorIncorrectIndex);
            return services::SharedPtr<FeatureIndicesRange>();
        }
        return services::internal::wrapSharedAndTryThrow<FeatureIndicesRange>(new FeatureIndicesRange(begin, end), status);
    }

    virtual size_t size() const override { return services::internal::maxValue(_begin, _end) - services::internal::minValue(_begin, _end) + 1; }

    virtual bool isPlainRange() const override { return true; }

    virtual bool areRawFeatureIndicesAvailable() const override { return false; }

    virtual FeatureIndex getFirst() const override { return _begin; }

    virtual FeatureIndex getLast() const override { return _end; }

    virtual services::BufferView<FeatureIndex> getRawFeatureIndices() override { return services::BufferView<FeatureIndex>(); }

private:
    explicit FeatureIndicesRange(FeatureIndex begin, FeatureIndex end) : _begin(begin), _end(end) {}

    FeatureIndex _begin;
    FeatureIndex _end;
};
typedef services::SharedPtr<FeatureIndicesList> FeatureIndicesListPtr;

} // namespace internal
} // namespace features
} // namespace data_management
} // namespace daal

#endif
