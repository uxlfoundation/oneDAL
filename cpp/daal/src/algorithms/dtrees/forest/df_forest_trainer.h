/* file: df_forest_trainer.h */
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
//  Forest layer of decision forest training on the tree engine (design 4.11):
//  per-tree engine streams, bootstrap, the tree engine, OOB error and
//  variable importance.
//--
*/

#ifndef __DF_FOREST_TRAINER_H__
#define __DF_FOREST_TRAINER_H__

#include "algorithms/decision_forest/decision_forest_training_parameter.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/engine/numa_alloc.h"
#include "src/algorithms/dtrees/forest/df_hyperparameter_impl.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training
{
namespace internal
{
using daal::internal::CpuType;

// Thread-seconds a fit takes per unit of work (trees x in-bag rows x drawn features x levels): 4.2-7.4e-8 on
// 2 x SPR 8480+ over letter to Higgs 10.5M (design 4.9). Provisional; 5e measures it.
constexpr double fitSecondsPerWorkUnit = 5.5e-8;

// Fits estimated below this use one NUMA node; that is every fit. On 2 x SPR 8480+ NUMA arenas were neutral to 19%
// slower at every size measured (scikit-learn_bench ensemble cases to Higgs 10.5M), and the first topology query
// costs 65-100 ms per process. The gate stays for hosts with more sockets or SNC, behind a measurement (design 4.9).
constexpr double numaMinFitSeconds = 1e300;

/// What a fit's time scales with.
struct FitWork
{
    size_t nTrees          = 0;
    size_t rows            = 0; // distinct in-bag rows of a tree
    size_t featuresPerNode = 0;
    size_t levels          = 0;
};

/// The work of a fit from its parameters; featuresPerNode at most the feature count, maxDepth and maxLeafNodes 0 when unbounded.
template <CpuType cpu>
inline FitWork fitWork(size_t nTrees, size_t nRows, size_t nSamples, bool bootstrap, size_t featuresPerNode, size_t minLeafRows, size_t maxDepth,
                       size_t maxLeafNodes)
{
    // Bootstrap counts are weights, so a tree's rows are its distinct draws, about 1 - 1/e of them.
    const size_t drawn = nSamples < nRows ? nSamples : nRows;
    const size_t rows  = bootstrap ? size_t(double(drawn) * 0.632) + 1 : drawn;
    size_t levels      = 0;
    for (size_t r = rows / (minLeafRows ? minLeafRows : 1); r > 1; r >>= 1) ++levels;
    if (maxDepth && maxDepth < levels) levels = maxDepth;
    size_t leafLevels = 0;
    for (size_t m = maxLeafNodes; m > 1; m >>= 1) ++leafLevels;
    if (maxLeafNodes && leafLevels < levels) levels = leafLevels;
    return { nTrees, rows, featuresPerNode, levels ? levels : 1 };
}

template <CpuType cpu>
inline double estimatedFitSeconds(const FitWork & work, size_t nThreads)
{
    const double units = double(work.nTrees) * double(work.rows) * double(work.featuresPerNode) * double(work.levels);
    return fitSecondsPerWorkUnit * units / double(nThreads ? nThreads : 1);
}

/// One node for fits estimated below numaMinFitSeconds, without querying the topology; else every node of the host.
template <CpuType cpu>
inline dtrees::engine::internal::NumaPlacement numaPlacement(const FitWork & work, size_t nThreads)
{
    using dtrees::engine::internal::NumaPlacement;
    return estimatedFitSeconds<cpu>(work, nThreads) < numaMinFitSeconds ? NumaPlacement() : NumaPlacement::host<cpu>();
}

/// Result tables of a fit, allocated by the caller; null when not requested.
struct ForestResults
{
    data_management::NumericTable * varImportance            = nullptr; // 1 x nFeatures
    data_management::NumericTable * oobError                 = nullptr; // 1 x 1
    data_management::NumericTable * oobErrorPerObservation   = nullptr; // nRows x 1
    data_management::NumericTable * oobErrorAccuracy         = nullptr; // 1 x 1, classification
    data_management::NumericTable * oobErrorR2               = nullptr; // 1 x 1, regression
    data_management::NumericTable * oobErrorDecisionFunction = nullptr; // nRows x nClasses, classification
    data_management::NumericTable * oobErrorPrediction       = nullptr; // nRows x 1, regression
    engines::EnginePtr updatedEngine;                                   // par.engine advanced past every draw of the fit
};

/**
 *  \brief Trains par.nTrees trees into model, which must be cleared
 *
 *  nClasses is 0 for regression. hist bins the table and runs the histogram phase where the tree engine
 *  chooses it; otherwise the exact phase everywhere (defaultDense).
 */
template <typename algorithmFPType, CpuType cpu>
services::Status trainForest(const data_management::NumericTable & x, const data_management::NumericTable & y,
                             const data_management::NumericTable * w, const Parameter & par, size_t nClasses, bool hist,
                             const decision_forest::internal::EngineHyperparameters & hp, dtrees::internal::ModelImpl & model, ForestResults & res);

} // namespace internal
} // namespace training
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
