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
/// Scoped as `enum class` so unqualified `bruteForceDense` / `kdTree` /
/// `ballTree` names cannot leak into user code.
enum class Method
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
/// runs the full pipeline: pairwise/core distances -> MST under MRD -> sort ->
/// dendrogram -> condensed tree -> cluster selection -> label points.
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
    /// @param[in]  ntData                  Input numeric table of size `N x P` containing the data to cluster
    /// @param[out] ntAssignments           Output numeric table of size `N x 1` containing the cluster assignment
    ///                                     for each input point. -1 indicates noise; non-negative values are the
    ///                                     cluster index in `[0, C)`, where `C` is the number of clusters found
    /// @param[out] ntNClusters             Output numeric table of size `1 x 1` containing the number of clusters `C` found
    /// @param[out] ntProbabilities         Optional output numeric table of size `N x 1` containing the membership
    ///                                     strength of each point in the cluster it was assigned to, in `[0, 1]`.
    ///                                     Noise points get 0. Pass `nullptr` to skip the computation
    /// @param[out] ntSingleLinkageTree     Optional output numeric table of size `(N - 1) x 4` receiving the
    ///                                     single-linkage dendrogram, one row per merge in ascending distance
    ///                                     order: `[left, right, distance, size]`. Ids below `N` are original
    ///                                     points, id `N + k` is the cluster formed by row `k`. Pass `nullptr`
    ///                                     to skip it. Left untouched when `N < 2`, where there is no merge
    /// @param[in]  minClusterSize          Minimum number of points required to form a cluster
    /// @param[in]  minSamples              Number of neighbors used when computing core distances
    /// @param[in]  pairwiseDistance        Distance metric used for pairwise distances (see `algorithms::internal::PairwiseDistanceType`)
    /// @param[in]  minkowskiDegree         Exponent `p` for the Minkowski distance. Ignored for other metrics
    /// @param[in]  clusterSelection        Cluster selection strategy: 0 -- excess of mass, 1 -- leaf
    /// @param[in]  allowSingleCluster      If true, allow the root cluster of the condensed tree to be selected
    /// @param[in]  clusterSelectionEpsilon Distance threshold used to merge clusters closer than epsilon.
    ///                                     Kept as `double` at the public entry point to match the descriptor;
    ///                                     narrowed to `algorithmFPType` inside `applyClusterSelectionEpsilon`
    ///                                     before the per-cluster comparison so tight loops stay in a single
    ///                                     precision
    /// @param[in]  maxClusterSize          Maximum allowed cluster size (only used with cluster selection epsilon). 0 disables the limit
    /// @param[in]  alpha                   Robust single-linkage scaling factor (distances are divided by alpha)
    /// @param[in]  leafSize                Maximum number of points per leaf in the kd-tree / ball-tree. Ignored for brute force
    ///
    /// @return Status code
    services::Status compute(const NumericTable * ntData, NumericTable * ntAssignments, NumericTable * ntNClusters, NumericTable * ntProbabilities,
                             NumericTable * ntSingleLinkageTree, size_t minClusterSize, size_t minSamples,
                             algorithms::internal::PairwiseDistanceType pairwiseDistance = algorithms::internal::PairwiseDistanceType::euclidean,
                             double minkowskiDegree = 2.0, int clusterSelection = 0, bool allowSingleCluster = false,
                             double clusterSelectionEpsilon = 0.0, size_t maxClusterSize = 0, double alpha = 1.0, size_t leafSize = 40);
};

/// Cluster centers in scikit-learn's definitions, computed from a finished labeling.
///
/// @tparam algorithmFPType Floating-point type of the data
/// @tparam cpu             CPU dispatch tag
template <typename algorithmFPType, CpuType cpu>
class HDBSCANCentersKernel : public Kernel
{
public:
    /// Compute the probability-weighted centroid and/or the medoid of every cluster.
    ///
    /// The medoid of a cluster is the member `i` minimizing `sum_j dist(i, j) * weight_j` over its
    /// members `j` in the fitted metric; ties go to the lowest row. Empty clusters get zero rows.
    ///
    /// @param[in]  ntData           Input numeric table of size `N x P`
    /// @param[in]  ntAssignments    Cluster id per point, `N x 1`; -1 is noise
    /// @param[in]  ntWeights        Membership probability per point, `N x 1`
    /// @param[in]  nClusters        Number of clusters `C`
    /// @param[out] ntCentroids      Centroids, `C x P`; pass `nullptr` to skip
    /// @param[out] ntMedoids        Medoids, `C x P`; pass `nullptr` to skip
    /// @param[in]  pairwiseDistance Distance metric of the fit
    /// @param[in]  minkowskiDegree  Exponent `p` for the Minkowski distance
    ///
    /// @return Status code
    services::Status compute(const NumericTable * ntData, const NumericTable * ntAssignments, const NumericTable * ntWeights, size_t nClusters,
                             NumericTable * ntCentroids, NumericTable * ntMedoids, algorithms::internal::PairwiseDistanceType pairwiseDistance,
                             double minkowskiDegree);
};

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_KERNEL_H__
