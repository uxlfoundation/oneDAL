/* file: hdbscan_kernel.h */
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

#ifndef __HDBSCAN_KERNEL_H__
#define __HDBSCAN_KERNEL_H__

#include "src/algorithms/kernel.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/services/cpu_type.h"
#include "data_management/data/numeric_table.h"

using daal::data_management::NumericTable;
using daal::internal::CpuType;

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

/// Available methods of the HDBSCAN algorithm.
///
/// Kept in the internal src namespace because HDBSCAN is only exposed through
/// the oneDAL (oneAPI) interface; the legacy DAAL C++ API does not ship this
/// algorithm, so the method tag does not need to be part of the public API.
enum Method
{
    bruteForceDense = 0, ///< Brute-force method with full distance matrix
    kdTree          = 1, ///< K-d tree method: O(N log N) neighbor search, no N^2 distance matrix
    ballTree        = 2  ///< Ball tree method: hypersphere-based partitioning, robust to high dimensions
};

/// HDBSCAN batch kernel.
///
/// CPU-templated entry point dispatched by the oneAPI HDBSCAN compute kernel.
/// Each (algorithmFPType, method, cpu) instantiation lives in its own TU
/// (`hdbscan_{dense,kd_tree,ball_tree}_batch_fpt_cpu.cpp`). Member compute()
/// runs the full pipeline: pairwise/core distances → MST under MRD → sort →
/// dendrogram → condensed tree → cluster selection → label points.
///
/// @tparam algorithmFPType Floating-point type used for distances and lambdas
/// @tparam method          One of `bruteForceDense`, `kdTree`, `ballTree`
/// @tparam cpu             CPU dispatch tag
template <typename algorithmFPType, Method method, CpuType cpu>
class HDBSCANBatchKernel : public Kernel
{
public:
    /// Compute HDBSCAN clustering for the specified input data and parameters.
    ///
    /// @param[in]  ntData                  Input numeric table of size `N × P` containing the data to cluster
    /// @param[out] ntAssignments           Output numeric table of size `N × 1` containing the cluster assignment
    ///                                     for each input point. -1 indicates noise; non-negative values are the
    ///                                     cluster index in `[0, C)`, where `C` is the number of clusters found
    /// @param[out] ntNClusters             Output numeric table of size `1 × 1` containing the number of clusters `C` found
    /// @param[in]  minClusterSize          Minimum number of points required to form a cluster
    /// @param[in]  minSamples              Number of neighbors used when computing core distances
    /// @param[in]  pairwiseDistance        Distance metric used for pairwise distances (see `algorithms::internal::PairwiseDistanceType`)
    /// @param[in]  minkowskiDegree         Exponent `p` for the Minkowski distance. Ignored for other metrics
    /// @param[in]  clusterSelection        Cluster selection strategy: 0 — excess of mass, 1 — leaf
    /// @param[in]  allowSingleCluster      If true, allow the root cluster of the condensed tree to be selected
    /// @param[in]  clusterSelectionEpsilon Distance threshold used to merge clusters closer than epsilon
    /// @param[in]  maxClusterSize          Maximum allowed cluster size (only used with cluster selection epsilon). 0 disables the limit
    /// @param[in]  alpha                   Robust single-linkage scaling factor (distances are divided by alpha)
    /// @param[in]  leafSize                Maximum number of points per leaf in the kd-tree / ball-tree. Ignored for brute force
    ///
    /// @return Status code
    services::Status compute(const NumericTable * ntData, NumericTable * ntAssignments, NumericTable * ntNClusters, size_t minClusterSize,
                             size_t minSamples,
                             algorithms::internal::PairwiseDistanceType pairwiseDistance = algorithms::internal::PairwiseDistanceType::euclidean,
                             double minkowskiDegree = 2.0, int clusterSelection = 0, bool allowSingleCluster = false,
                             double clusterSelectionEpsilon = 0.0, size_t maxClusterSize = 0, double alpha = 1.0, size_t leafSize = 40);
};

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_KERNEL_H__
