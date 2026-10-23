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

// Tables of X below this size use one NUMA node. They sit in one socket's L3 (105 MB on SPR), so NUMA placement
// gains them nothing, while querying the topology costs 45-70 ms once per process on a 2-socket host and the NUMA
// arenas a few ms a fit (design 8, steps 5e and 15). Provisional; 5e measures it.
constexpr size_t numaMinTableBytes = size_t(32) << 20;

/// One node for tables of X below numaMinTableBytes, without querying the topology; else every node of the host.
inline dtrees::engine::internal::NumaPlacement numaPlacement(size_t tableBytes, bool replicate)
{
    using dtrees::engine::internal::NumaPlacement;
    return tableBytes < numaMinTableBytes ? NumaPlacement() : NumaPlacement::host(replicate);
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
