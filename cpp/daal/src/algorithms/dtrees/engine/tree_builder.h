/* file: tree_builder.h */
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

/*
//++
//  Builds decision trees with the exact phase at every node: the defaultDense
//  method (design 4.7). The forest supplies each tree's records and receives its
//  tables through TreeInputs.
//--
*/

#ifndef __DTREES_ENGINE_TREE_BUILDER_H__
#define __DTREES_ENGINE_TREE_BUILDER_H__

#include <cstddef>
#include <cstdint>

#include "data_management/data/homogen_numeric_table.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/engine/bin_matrix.h"
#include "src/algorithms/dtrees/engine/executor.h"
#include "src/algorithms/dtrees/engine/numa_alloc.h"
#include "src/algorithms/dtrees/engine/response_set.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace engine
{
namespace internal
{
using daal::internal::CpuType;

/// Tree-growing parameters, resolved once per fit from the public Parameter.
struct TreeParams
{
    size_t nClasses            = 0; // 0 for regression
    size_t featuresPerNode     = 0; // k features drawn per node; 0 or more than the features: all
    size_t maxDepth            = 0; // 0: unlimited
    size_t maxLeafNodes        = 0; // 0: depth-first; otherwise best-first with at most this many leaves
    size_t minLeafRecords      = 1; // records (in-bag rows) on each side of a split
    size_t minSplitRecords     = 2; // records a node needs to be split
    double minLeafWeight       = 0; // weight on each side of a split
    double minWeightedDecrease = 0; // a split's weighted impurity decrease must reach this
    double impurityThreshold   = 0; // nodes with impurity at or below this are leaves
    bool randomSplitter        = false;
    bool counterDraws          = true; // philox4x32x10 node draws by counter; otherwise TreeInputs::nodeStream
    size_t serialSubtreeRows   = 0;    // see ExecutorParams
    size_t treesInFlight       = 0;
    // hist: nodes of fewer than denseThreshold * maxBins records take the exact phase (design 4.7).
    size_t denseThreshold = 2;
    size_t maxBins        = 256;
    // Large histogram-phase nodes run in chunks of chunkRows records, at most maxChunks (design 4.6).
    size_t chunkRows = size_t(1) << 14;
    size_t maxChunks = 64;
};

/// The training data, read-only for the whole fit.
template <typename algorithmFPType, CpuType cpu>
struct TreeData
{
    const NumaReplicated<algorithmFPType, cpu> * x         = nullptr; // row-major, nRows x nFeatures, per NUMA node
    const ResponseSource<algorithmFPType, cpu> * responses = nullptr;
    const bool * categorical                               = nullptr; // per feature; nullptr: all ordered
    const BinMatrix<cpu> * bins                            = nullptr; // hist: the binned table; nullptr for defaultDense
    size_t nRows                                           = 0;
    size_t nFeatures                                       = 0;
};

/// The tables of one tree in the serialized DecisionTreeTable format (design 4.14).
struct EmittedTree
{
    dtrees::internal::DecisionTreeTablePtr nodes;
    services::SharedPtr<data_management::HomogenNumericTable<double> > impurity;
    services::SharedPtr<data_management::HomogenNumericTable<int> > nodeSamples;
    services::SharedPtr<data_management::HomogenNumericTable<double> > probabilities; // nClasses per node; empty for regression
    services::internal::TArray<double, DAAL_BASE_CPU> featureImportance;              // MDI: weighted decrease per feature
};

/// What the forest supplies per tree and receives back. Calls for different trees run concurrently.
template <typename Record, CpuType cpu>
class TreeInputs
{
public:
    virtual ~TreeInputs() {}
    /// Fills the tree's records (buffer 0 of records) and sets its seed, the root's node key.
    virtual services::Status prepareTree(size_t iTree, int numaNode, ResponseSet<Record, cpu> & records, uint64_t & treeSeed) = 0;
    /// Takes the finished tree.
    virtual services::Status emitTree(size_t iTree, EmittedTree & tree) = 0;
    /// The calling thread's stream of the user's engine, for TreeParams::counterDraws == false.
    virtual void * nodeStream() { return nullptr; }
};

/**
 *  \brief Builds nTrees trees on the executor
 *
 *  Record is a response_set.h layout: ClsCountRecord or ClsWeightRecord for classification (Gini),
 *  RegCountRecord or RegWeightRecord for regression (MSE).
 */
template <typename algorithmFPType, typename Record, CpuType cpu>
services::Status trainTrees(const TreeData<algorithmFPType, cpu> & data, const TreeParams & params, size_t nTrees, TreeInputs<Record, cpu> & inputs);

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
