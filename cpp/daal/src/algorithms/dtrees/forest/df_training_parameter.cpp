/* file: df_training_parameter.cpp */
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
//  Implementation of decision forest training parameter class
//--
*/

#include "algorithms/decision_forest/decision_forest_training_parameter.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training
{
namespace interface2
{
using namespace daal::services;

Parameter::Parameter()
    : nTrees(100),
      observationsPerTreeFraction(1.),
      featuresPerNode(0),
      maxTreeDepth(0),
      minObservationsInLeafNode(0),
      seed(777),
      engine(engines::mt2203::Batch<>::create()),
      impurityThreshold(0.),
      varImportance(none),
      resultsToCompute(0),
      memorySavingMode(false),
      bootstrap(true),
      minObservationsInSplitNode(2),
      minWeightFractionInLeafNode(0.),
      minImpurityDecreaseInSplitNode(0.),
      maxLeafNodes(0),
      minBinSize(5),
      maxBins(256),
      splitter(best),
      binningStrategy(quantiles)
{}
} // namespace interface2
Status checkImpl(const decision_forest::training::interface2::Parameter & prm)
{
    DAAL_CHECK_EX(prm.nTrees, ErrorIncorrectParameter, ParameterName, nTreesStr());
    DAAL_CHECK_EX(prm.minObservationsInLeafNode, ErrorIncorrectParameter, ParameterName, minObservationsInLeafNodeStr());
    DAAL_CHECK_EX(prm.minObservationsInSplitNode > 1, ErrorIncorrectParameter, ParameterName, minObservationsInSplitNodeStr());
    DAAL_CHECK_EX((prm.minWeightFractionInLeafNode >= 0.0) && (prm.minWeightFractionInLeafNode <= 0.5), ErrorIncorrectParameter, ParameterName,
                  minWeightFractionInLeafNodeStr());
    DAAL_CHECK_EX(prm.minImpurityDecreaseInSplitNode >= 0.0, ErrorIncorrectParameter, ParameterName, minImpurityDecreaseInSplitNodeStr());
    DAAL_CHECK_EX((prm.observationsPerTreeFraction > 0) && (prm.observationsPerTreeFraction <= 1), ErrorIncorrectParameter, ParameterName,
                  observationsPerTreeFractionStr());
    DAAL_CHECK_EX((prm.impurityThreshold >= 0), ErrorIncorrectParameter, ParameterName, impurityThresholdStr());
    Status s;
    if (!prm.bootstrap)
    {
        if (prm.varImportance == MDA_Raw || prm.varImportance == MDA_Scaled) s.add(Error::create(ErrorDFBootstrapVarImportanceIncompatible));
        if (prm.resultsToCompute & computeOutOfBagError) s.add(Error::create(ErrorDFBootstrapOOBIncompatible));
    }
    DAAL_CHECK_EX((prm.minBinSize >= 1), ErrorIncorrectParameter, ParameterName, minBinSizeStr());
    return s;
}
} // namespace training
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
