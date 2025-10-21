/* file: decision_forest_training_parameter.h */
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
//  Decision forest training parameter class
//--
*/

#ifndef __DECISION_FOREST_TRAINING_PARAMETER_H__
#define __DECISION_FOREST_TRAINING_PARAMETER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/engines/mt2203/mt2203.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup base_decision_forest Base Decision Forest
 * \brief Contains base classes of the decision forest algorithm
 * @ingroup training_and_prediction
 */
/**
 * \brief Contains classes of the decision forest algorithm
 */
namespace decision_forest
{
/**
 * \brief Contains a class for decision forest model-based training
 */
namespace training
{
/**
 * @ingroup base_decision_forest
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__TRAINING__VARIABLE_IMPORTANCE_MODE"></a>
 * \brief Variable importance computation mode
 */
enum VariableImportanceMode
{
    none,      /* Do not compute */
    MDI,       /* Mean Decrease Impurity.
                       Computed as the sum of weighted impurity decreases for all nodes where the variable is used,
                       averaged over all trees in the forest */
    MDA_Raw,   /* Mean Decrease Accuracy (permutation importance).
                       For each tree, the prediction error on the out-of-bag portion of the data is computed
                       (error rate for classification, MSE for regression).
                       The same is done after permuting each predictor variable.
                       The difference between the two are then averaged over all trees. */
    MDA_Scaled /* Mean Decrease Accuracy (permutation importance).
                       This is MDA_Raw value scaled by its standard deviation. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__TRAINING__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the result to compute
 */
enum ResultToComputeId
{
    computeOutOfBagError                 = 0x00000001ULL,
    computeOutOfBagErrorPerObservation   = 0x00000002ULL,
    computeOutOfBagErrorAccuracy         = 0x00000004ULL,
    computeOutOfBagErrorR2               = 0x00000008ULL,
    computeOutOfBagErrorDecisionFunction = 0x00000010ULL,
    computeOutOfBagErrorPrediction       = 0x00000020ULL
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DECISION_FOREST__TRAINING__SPLITTER_MODE"></a>
 * \brief Node splitting mode
 */
enum SplitterMode
{
    best,  /* Calculates best split from aggregate best feature splits for every node. */
    random /* Calculates best split from aggregate random feature splits for every node. */
};

/**
 * <a name = "DAAL-ENUM-ALGORITHMS__DECISION_FOREST__TRAINING__BINNINGSTRATEGY"></a>
 * \brief Available strategies to compute data bins in 'hist' method
 */
enum BinningStrategy
{
    /* Frequency quantiles -> same number of data points per bin */
    quantiles,
    /* Same feature value range per bin */
    averages
};

/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__TRAINING__PARAMETER"></a>
 * \brief Parameters for the decision forest algorithm
 *
 * \snippet decision_forest/decision_forest_training_parameter.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter
{
public:
    /**
     * Construct parameters of decision forest algorithm
     */
    Parameter();

    size_t nTrees;                         /*!< Number of trees in the forest. Default is 10 */
    double observationsPerTreeFraction;    /*!< Fraction of observations used for a training of one tree, 0 to 1.
                                                  Default is 1 (sampling with replacement) */
    size_t featuresPerNode;                /*!< Number of features tried as possible splits per node.
                                                  If 0 then sqrt(p) for classification, p/3 for regression,
                                                  where p is the total number of features. */
    size_t maxTreeDepth;                   /*!< Maximal tree depth. Default is 0 (unlimited) */
    size_t minObservationsInLeafNode;      /*!< Minimal number of observations in a leaf node.
                                                  Default is 1 for classification, 5 for regression. */
    size_t seed;                           /*!< Seed for the random numbers generator used by the algorithms \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine;             /*!< Engine for the random numbers generator used by the algorithms */
    double impurityThreshold;              /*!< Threshold value used as stopping criteria: if the impurity value in the node is smaller
                                                  than the threshold then the node is not split anymore.*/
    VariableImportanceMode varImportance;  /*!< Variable importance computation mode */
    DAAL_UINT64 resultsToCompute;          /*!< 64 bit integer flag that indicates the results to compute */
    bool memorySavingMode;                 /*!< If true then use memory saving (but slower) mode */
    bool bootstrap;                        /*!< If true then training set for a tree is a bootstrap of the whole training set */
    size_t minObservationsInSplitNode;     /*!< Minimal number of observations in a split node. Default 2 */
    double minWeightFractionInLeafNode;    /*!< The minimum weighted fraction of the sum total of weights (of all the input observations)
                                                  required to be at a leaf node, 0.0 to 0.5. Default is 0.0 */
    double minImpurityDecreaseInSplitNode; /*!< A node will be split if this split induces a decrease of the impurity
                                                  greater than or equal to the value, non-negative. Default is 0.0 */
    size_t maxLeafNodes;                   /*!< Maximum number of leaf node. Default is 0 (unlimited) */
    size_t maxBins;                        /*!< Used with 'hist' split finding method only.
                                                 Maximal number of discrete bins to bucket continuous features.
                                                 Default is 256. Increasing the number results in higher computation costs */
    size_t minBinSize;                     /*!< Used with 'hist' split finding method only.
                                                 Minimal number of observations in a bin. Default is 5 */
    SplitterMode splitter;                 /*!< Sets node splitting method. Default is best */
    BinningStrategy binningStrategy;       /*!< Used with 'hist' split finding method only.
                                                 Selects the strategy to group data points into bins.
                                                 Allowed values are 'quantiles' (default), 'averages' */
};
/* [Parameter source code] */
} // namespace interface2
using interface2::Parameter;
/** @} */
} // namespace training
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
#endif
