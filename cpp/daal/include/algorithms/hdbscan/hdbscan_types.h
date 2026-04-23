/* file: hdbscan_types.h */
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

#ifndef __HDBSCAN_TYPES_H__
#define __HDBSCAN_TYPES_H__

namespace daal
{
namespace algorithms
{
namespace hdbscan
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__HDBSCAN__METHOD"></a>
 * Available methods of the HDBSCAN algorithm
 */
enum Method
{
    defaultDense = 0, /*!< Default: brute-force method with full distance matrix */
    kdTree       = 1  /*!< K-d tree method: O(N log N) neighbor search, no N^2 distance matrix */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__HDBSCAN__METRIC"></a>
 * Available distance metrics for the HDBSCAN algorithm
 */
enum Metric
{
    euclidean = 0, /*!< Euclidean (L2) distance */
    manhattan = 1, /*!< Manhattan (L1) distance */
    minkowski = 2, /*!< Minkowski (Lp) distance with configurable degree */
    chebyshev = 3, /*!< Chebyshev (L-infinity) distance */
    cosine    = 4  /*!< Cosine distance (1 - cosine_similarity) */
};

} // namespace hdbscan
} // namespace algorithms
} // namespace daal

#endif // __HDBSCAN_TYPES_H__
