/* file: df_classification_train_impl.i */
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
//  Decision forest classification training: the forest layer on the tree engine
//  (df_forest_trainer.i), or the legacy kernel when the legacyTraining
//  hyperparameter is set.
//--
*/

#ifndef __DF_CLASSIFICATION_TRAIN_IMPL_I__
#define __DF_CLASSIFICATION_TRAIN_IMPL_I__

#include "src/algorithms/dtrees/forest/classification/df_classification_train_dense_default_impl.i"
#include "src/algorithms/dtrees/forest/df_forest_trainer.i"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status ClassificationTrainBatchKernel<algorithmFPType, method, cpu>::compute(
    const NumericTable * x, const NumericTable * y, const NumericTable * w, decision_forest::classification::Model & m, Result & res,
    const decision_forest::classification::training::Parameter & par, const HyperparameterType * hp)
{
    services::Status s;
    if (hp)
    {
        hp->check(s);
        DAAL_CHECK_STATUS_VAR(s);
    }
    decision_forest::internal::EngineHyperparameters engineHp;
    s = decision_forest::internal::readEngineHyperparameters(hp, denseThreshold, chunkRows, maxChunks, treesInFlight, legacyTraining, engineHp);
    DAAL_CHECK_STATUS_VAR(s);
    if (engineHp.legacyTraining) return computeLegacy<algorithmFPType, method, cpu>(x, y, w, m, res, par, hp);
    DAAL_CHECK(x && y, services::ErrorNullInputNumericTable);

    decision_forest::training::internal::ForestResults fr;
    if (par.varImportance != decision_forest::training::none) fr.varImportance = res.get(variableImportance).get();
    if (par.resultsToCompute & decision_forest::training::computeOutOfBagError) fr.oobError = res.get(outOfBagError).get();
    if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
        fr.oobErrorPerObservation = res.get(outOfBagErrorPerObservation).get();
    if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorAccuracy) fr.oobErrorAccuracy = res.get(outOfBagErrorAccuracy).get();
    if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorDecisionFunction)
        fr.oobErrorDecisionFunction = res.get(outOfBagErrorDecisionFunction).get();
    auto & model = *static_cast<decision_forest::classification::internal::ModelImpl *>(&m);
    s = decision_forest::training::internal::trainForest<algorithmFPType, cpu>(*x, *y, w, par, par.nClasses, method == hist, engineHp, model, fr);
    if (s) res.impl()->setEngine(fr.updatedEngine);
    return s;
}

} // namespace internal
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
