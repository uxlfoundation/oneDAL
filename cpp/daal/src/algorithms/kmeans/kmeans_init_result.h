/* file: kmeans_init_result.h */
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
//  Implementation of kmeans classes.
//--
*/

#ifndef __KMEANS_INIT_RESULT_
#define __KMEANS_INIT_RESULT_

#include "algorithms/kmeans/kmeans_init_types.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
/**
 * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
 * \param[in] input        Pointer to the input structure
 * \param[in] parameter    Pointer to the parameter structure
 * \param[in] method       Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const interface1::Parameter * kmPar             = static_cast<const interface1::Parameter *>(parameter);
    const DistributedStep2MasterInput * masterInput = dynamic_cast<const DistributedStep2MasterInput *>(input);
    size_t nFeatures                                = 0;
    if (masterInput)
    {
        data_management::DataCollection * coll = masterInput->get(partialResults).get();
        for (size_t i = 0; i < coll->size(); ++i)
        {
            data_management::NumericTable * partClusters = static_cast<PartialResult *>((*coll)[i].get())->get(partialClusters).get();
            if (partClusters)
            {
                nFeatures = partClusters->getNumberOfColumns();
                break;
            }
        }
    }
    else
        nFeatures = (static_cast<const Input *>(input))->get(data)->getNumberOfColumns();

    services::Status status;

    Argument::set(centroids, data_management::SerializationIfacePtr(new data_management::HomogenNumericTable<algorithmFPType>(
                                 nFeatures, kmPar->nClusters, data_management::NumericTable::doAllocate)));
    return status;
}

/**
 * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
 * \param[in] partialResult Pointer to the partial result structure
 * \param[in] parameter     Pointer to the parameter structure
 * \param[in] method        Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter,
                                              const int method)
{
    const interface1::Parameter * kmPar = static_cast<const interface1::Parameter *>(parameter);

    size_t nClusters                                          = kmPar->nClusters;
    size_t nFeatures                                          = 0;
    const DistributedStep5MasterPlusPlusPartialResult * step5 = dynamic_cast<const DistributedStep5MasterPlusPlusPartialResult *>(partialResult);
    if (step5)
        nFeatures = step5->get(candidates)->getNumberOfColumns();
    else
    {
        const data_management::NumericTable * pPartialClusters = static_cast<const PartialResult *>(partialResult)->get(partialClusters).get();
        if (pPartialClusters) nFeatures = pPartialClusters->getNumberOfColumns();
    }
    Argument::set(centroids, data_management::SerializationIfacePtr(new data_management::HomogenNumericTable<algorithmFPType>(
                                 nFeatures, nClusters, data_management::NumericTable::doAllocate)));
    return services::Status();
}

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
