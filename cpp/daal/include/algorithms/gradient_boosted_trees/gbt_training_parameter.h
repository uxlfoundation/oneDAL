/* file: gbt_training_parameter.h */
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
//  Gradient Boosted Trees training parameter class
//--
*/

#ifndef __GBT_TRAINING_PARAMETER_H__
#define __GBT_TRAINING_PARAMETER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/engines/engine.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup base_gbt Base Gradient Boosted Trees
 * \brief Contains base classes of the gradient boosted trees algorithm
 * @ingroup training_and_prediction
 */
/**
 * \brief Contains classes of the gradient boosted trees algorithm
 */
namespace gbt
{
/**
 * \brief Contains a class for model-based training
 */
namespace training
{
/**
 * @ingroup base_gbt
 * @{
 */

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__TRAINING__SPLIT_METHOD"></a>
 * \brief Split finding method in gradient boosted trees algorithm
 */
enum SplitMethod
{
    exact        = 0,      /*!< Exact greedy method */
    inexact      = 1,      /*!< Inexact method for splits finding: bucket continuous features to discrete bins */
    defaultSplit = inexact /*!< Default split finding method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__GBT__TRAINING__VARIABLE_IMPORTANCE_MODES"></a>
 * \brief Variable importance computation modes
 */
enum VariableImportanceModes
{
    weight     = 0x001ULL,
    totalCover = 0x002ULL,
    cover      = 0x004ULL,
    totalGain  = 0x008ULL,
    gain       = 0x010ULL
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__TRAINING__PARAMETER"></a>
 * \brief Parameters for the gradient boosted trees algorithm
 *
 * \snippet gradient_boosted_trees/gbt_training_parameter.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter
{
public:
    Parameter();

    SplitMethod splitMethod;            /*!< Split finding method. Default is exact */
    size_t maxIterations;               /*!< Maximal number of iterations of the gradient boosted trees training algorithm.
                                                 Default is 50 */
    size_t maxTreeDepth;                /*!< Maximal tree depth, 0 for unlimited. Default is 6 */
    double shrinkage;                   /*!< Learning rate of the boosting procedure.
                                                 Scales the contribution of each tree by a factor (0, 1].
                                                 Default is 0.3 */
    double minSplitLoss;                /*!< Loss regularization parameter. Min loss reduction required to make a further partition
                                                 on a leaf node of the tree.
                                                 Range: [0, inf). Default is 0 */
    double lambda;                      /*!< L2 regularization parameter on weights.
                                                 Range: [0, inf). Default is 1 */
    double observationsPerTreeFraction; /*!< Fraction of observations used for a training of one tree, sampling without replacement.
                                                 Range: (0, 1]. Default is 1 (no sampling, entire dataset is used) */
    size_t featuresPerNode;             /*!< Number of features tried as possible splits per node.
                                                 Range : [0, p] where p is the total number of features.
                                                 Default is 0 (use all features) */
    size_t minObservationsInLeafNode;   /*!< Minimal number of observations in a leaf node. Default is 5. */
    bool memorySavingMode;              /*!< If true then use memory saving (but slower) mode. Default is false */
    engines::EnginePtr engine;          /*!< Engine for the random numbers generator used by the algorithms */
    size_t maxBins;                     /*!< Used with 'inexact' split finding method only.
                                                 Maximal number of discrete bins to bucket continuous features.
                                                 Default is 256. Increasing the number results in higher computation costs */
    size_t minBinSize;                  /*!< Used with 'inexact' split finding method only.
                                                 Minimal number of observations in a bin. Default is 5 */
    int internalOptions;                /*!< Internal options */
};
/* [Parameter source code] */
} // namespace interface1
using interface1::Parameter;
/** @} */
} // namespace training
} // namespace gbt
} // namespace algorithms
} // namespace daal
#endif
