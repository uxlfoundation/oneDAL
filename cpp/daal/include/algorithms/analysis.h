/* file: analysis.h */
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
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__

#include "algorithms/algorithm_base.h"

namespace daal
{
/**
 * @defgroup algorithms Algorithms
 * @{
 */
namespace algorithms
{
/**
 * @defgroup analysis Analysis
 * \brief Contains classes for analysis algorithms that are intended to uncover the underlying structure
 *        of a data set and to characterize it by a set of quantitative measures, such as statistical moments,
 *        correlations coefficients, and so on.
 * @ingroup algorithms
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSISCONTAINERIFACE"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the analysis algorithms. It is associated with the Analysis class
 *        and supports the methods for computation and finalization of the analysis results
 *        in the batch, distributed, and online modes.
 *        The methods of the container are defined in derivative containers
 *        defined for each algorithm of data analysis.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template <ComputeMode mode>
class AnalysisContainerIface : public AlgorithmContainerImpl<mode>
{
public:
    AnalysisContainerIface(daal::services::Environment::env * daalEnv = 0) : AlgorithmContainerImpl<mode>(daalEnv) {}

    AnalysisContainerIface(const AnalysisContainerIface &)                               = delete;
    AnalysisContainerIface<mode> & operator=(const AnalysisContainerIface<mode> & other) = delete;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ANALYSIS"></a>
 * \brief Provides methods for execution of operations over data, such as computation of Summary Statistics estimates.
 *        The methods of the class support different computation modes: batch, distributed, and online(see \ref ComputeMode).
 *        Classes that implement specific algorithms of the data analysis are derived classes of the \ref Analysis class.
 *        The class additionally provides virtual methods for validation of input and output parameters
 *        of the algorithms.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template <ComputeMode mode>
class Analysis : public AlgorithmImpl<mode>
{};
} // namespace algorithms
} // namespace daal
/** @} */
/** @} */
#endif
