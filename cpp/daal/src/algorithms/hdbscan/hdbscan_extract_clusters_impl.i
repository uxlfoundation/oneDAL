/* file: hdbscan_extract_clusters_impl.i */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <cstdint>

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/hdbscan/hdbscan_cluster_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_arrays.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

using daal::internal::CpuType;
using daal::internal::ReadRows;
using daal::internal::WriteOnlyRows;
using daal::services::internal::TArray;

template <typename algorithmFPType, CpuType cpu>
services::Status HDBSCANExtractClustersKernel<algorithmFPType, cpu>::compute(const NumericTable * ntMstFrom, const NumericTable * ntMstTo,
                                                                             const NumericTable * ntMstWeights, size_t nRows,
                                                                             NumericTable * ntAssignments, NumericTable * ntNClusters,
                                                                             NumericTable * ntProbabilities, NumericTable * ntSingleLinkageTree,
                                                                             size_t minClusterSize, int clusterSelection, bool allowSingleCluster,
                                                                             double clusterSelectionEpsilon, size_t maxClusterSize)
{
    DAAL_CHECK(nRows > 0, services::ErrorEmptyInputNumericTable);
    // Labels are stored as int32, and there are at most nRows / minClusterSize clusters.
    DAAL_CHECK(minClusterSize > 0 && nRows / minClusterSize <= static_cast<size_t>(INT32_MAX), services::ErrorIncorrectSizeOfInputNumericTable);
    const size_t edgeCount = nRows - 1;

    // The extraction sorts the edges in place, so it works on copies.
    TArray<DAAL_INT, cpu> mstFromArr(edgeCount);
    TArray<DAAL_INT, cpu> mstToArr(edgeCount);
    TArray<algorithmFPType, cpu> mstWeightsArr(edgeCount);
    DAAL_INT * mstFrom           = mstFromArr.get();
    DAAL_INT * mstTo             = mstToArr.get();
    algorithmFPType * mstWeights = mstWeightsArr.get();
    if (edgeCount > 0)
    {
        DAAL_CHECK_MALLOC(mstFrom && mstTo && mstWeights);

        ReadRows<int, cpu> fromRows(const_cast<NumericTable *>(ntMstFrom), 0, edgeCount);
        DAAL_CHECK_BLOCK_STATUS(fromRows);
        ReadRows<int, cpu> toRows(const_cast<NumericTable *>(ntMstTo), 0, edgeCount);
        DAAL_CHECK_BLOCK_STATUS(toRows);
        ReadRows<algorithmFPType, cpu> weightRows(const_cast<NumericTable *>(ntMstWeights), 0, edgeCount);
        DAAL_CHECK_BLOCK_STATUS(weightRows);

        const int * from               = fromRows.get();
        const int * to                 = toRows.get();
        const algorithmFPType * weight = weightRows.get();
        for (size_t e = 0; e < edgeCount; e++)
        {
            DAAL_CHECK(from[e] >= 0 && static_cast<size_t>(from[e]) < nRows && to[e] >= 0 && static_cast<size_t>(to[e]) < nRows,
                       services::ErrorIncorrectIndex);
            mstFrom[e]    = static_cast<DAAL_INT>(from[e]);
            mstTo[e]      = static_cast<DAAL_INT>(to[e]);
            mstWeights[e] = weight[e];
        }
        const services::Status mstStatus = checkMst(edgeCount, edgeCount, mstWeights);
        DAAL_CHECK_STATUS_VAR(mstStatus);
    }

    WriteOnlyRows<int, cpu> assignBlock(ntAssignments, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(assignBlock);

    WriteOnlyRows<algorithmFPType, cpu> probBlock;
    algorithmFPType * probabilities = probBlock.set(ntProbabilities, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(probBlock);

    WriteOnlyRows<algorithmFPType, cpu> sltBlock;
    algorithmFPType * singleLinkageTree = (edgeCount > 0) ? sltBlock.set(ntSingleLinkageTree, 0, edgeCount) : nullptr;
    DAAL_CHECK_BLOCK_STATUS(sltBlock);

    int labelCounter        = 0;
    services::Status status = sortMstAndExtractClusters<algorithmFPType, cpu>(
        mstFrom, mstTo, mstWeights, nRows, minClusterSize, assignBlock.get(), labelCounter, clusterSelection, allowSingleCluster,
        clusterSelectionEpsilon, maxClusterSize, probabilities, singleLinkageTree);
    DAAL_CHECK_STATUS_VAR(status);

    WriteOnlyRows<int, cpu> ncBlock(ntNClusters, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(ncBlock);
    ncBlock.get()[0] = labelCounter;

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
