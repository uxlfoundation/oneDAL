/* file: bf_knn_classification_model_impl.h */
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

#ifndef __BF_KNN_CLASSIFICATION_MODEL_IMPL_
#define __BF_KNN_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace interface1
{
class Model::ModelImpl
{
public:
    ModelImpl(size_t nFeatures = 0) : _nFeatures(nFeatures) {}

    data_management::NumericTableConstPtr getData() const { return _data; }

    data_management::NumericTablePtr getData() { return _data; }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_nFeatures);
        arch->setSharedPtrObj(_data);
        arch->setSharedPtrObj(_labels);

        return services::Status();
    }

    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setData(const data_management::NumericTablePtr & value, bool copy)
    {
        return setTable<algorithmFPType>(value, _data, copy);
    }

    data_management::NumericTableConstPtr getLabels() const { return _labels; }

    data_management::NumericTablePtr getLabels() { return _labels; }

    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setLabels(const data_management::NumericTablePtr & value, bool copy)
    {
        return setTable<algorithmFPType>(value, _labels, copy);
    }

    size_t getNumberOfFeatures() const { return _nFeatures; }

protected:
    template <typename algorithmFPType>
    DAAL_FORCEINLINE services::Status setTable(const data_management::NumericTablePtr & value, data_management::NumericTablePtr & dest, bool copy)
    {
        if (!copy)
        {
            dest = value;
        }
        else
        {
            services::Status status;
            dest = data_management::HomogenNumericTable<algorithmFPType>::create(value->getNumberOfColumns(), value->getNumberOfRows(),
                                                                                 data_management::NumericTable::doAllocate, &status);
            DAAL_CHECK_STATUS_VAR(status);
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            DAAL_CHECK_STATUS_VAR(dest->getBlockOfRows(0, dest->getNumberOfRows(), data_management::writeOnly, destBD));
            DAAL_CHECK_STATUS_VAR(value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD));
            services::internal::daal_memcpy_s(destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType),
                                              srcBD.getBlockPtr(), srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            DAAL_CHECK_STATUS_VAR(dest->releaseBlockOfRows(destBD));
            DAAL_CHECK_STATUS_VAR(value->releaseBlockOfRows(srcBD));
        }
        return services::Status();
    }

private:
    size_t _nFeatures;
    data_management::NumericTablePtr _data;
    data_management::NumericTablePtr _labels;
};

} // namespace interface1
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
