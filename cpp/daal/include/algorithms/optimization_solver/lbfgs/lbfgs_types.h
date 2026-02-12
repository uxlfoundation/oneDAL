/* file: lbfgs_types.h */
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
//  Implementation of limited memory Broyden-Fletcher-Goldfarb-Shanno
//  algorithm types.
//--
*/
#ifndef __LBFGS_TYPES_H__
#define __LBFGS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/memory_block.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_batch.h"
#include "algorithms/engines/mt19937/mt19937.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup optimization_solver Optimization Solvers
 * \copydoc daal::algorithms::optimization_solver
 * @ingroup analysis
 */
/**
 * \brief Contains classes for optimization solver algorithms
 */
namespace optimization_solver
{
/**
 * @defgroup lbfgs Limited-Memory-Broyden-Fletcher-Goldfarb-Shanno Algorithm
 * \copydoc daal::algorithms::optimization_solver::lbfgs
 * @ingroup optimization_solver
 * @{
 */
/**
 * \brief Contains classes for computing the limited memory Broyden-Fletcher-Goldfarb-Shanno algorithm
 */
namespace lbfgs
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__METHOD"></a>
 * Available methods for computing LBFGS
 * @ingroup lbfgs
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__LBFGS__OPTIONALDATAID"></a>
* Available identifiers of optional input for the iterative solver
*/
enum OptionalDataId
{
    correctionPairs = iterative_solver::lastOptionalData + 1, /*!< Correction pairs table. Numeric table 2*m x n, where
                                         rows (0, m-1) represent correction vectors S and
                                         rows (m, 2*m-1) represent correction vectors Y */
    correctionIndices,                                        /*!< Numeric table of size 1 x 2 with 32-bit integer indexes.
                                         The first value is the index of correction pair t,
                                         the second value is the index of the last iteration k from the previous run */
    averageArgumentLIterations,                               /*!< Numeric table of size 2 x n, where
                                         row 0 represent average arguments for the previous L iterations and
                                         row 1 represent average arguments for the last L iterations.
                                         These values are required to compute S correction vectors on the next step */
    lastOptionalData = averageArgumentLIterations
};

namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__PARAMETER"></a>
 * \brief %Parameter class for LBFGS algorithm
 *
 * \snippet optimization_solver/lbfgs/lbfgs_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public optimization_solver::iterative_solver::Parameter
{
    /**
     * Constructs the parameters of LBFGS algorithm
     * \param[in] function                  Objective function that can be represented as sum
     * \param[in] nIterations               Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold         Accuracy of the LBFGS algorithm
     * \param[in] batchSize                 Number of observations to compute the stochastic gradient
     * \param[in] correctionPairBatchSize_  The number of observations to compute the sub-sampled Hessian for correction pairs computation
     * \param[in] m                         Memory parameter of LBFGS
     * \param[in] L                         The number of iterations between the curvature estimates calculations
     * \param[in] seed                      Seed for random choosing terms from objective function \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(sum_of_functions::BatchPtr function = sum_of_functions::BatchPtr(), size_t nIterations = 100, double accuracyThreshold = 1.0e-5,
              size_t batchSize = 10, size_t correctionPairBatchSize_ = 100, size_t m = 10, size_t L = 10, size_t seed = 777);

    virtual ~Parameter() {}

    size_t m;                  /*!< Memory parameter of LBFGS.
                                         The maximum number of correction pairs that define the approximation
                                         of inverse Hessian matrix. */
    size_t L;                  /*!< The number of iterations between the curvature estimates calculations */
    size_t seed;               /*!< Seed for random choosing terms from objective function. \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine; /*!< Engine for random choosing terms from objective function. */

    data_management::NumericTablePtr batchIndices;

    size_t correctionPairBatchSize; /*!< Number of observations to compute the sub-sampled Hessian for correction pairs computation */
    /** Numeric table of size (nIterations / L) x correctionPairBatchSize that represent indices that will be used
        instead of random values for the sub-sampled Hessian matrix computations. If not set then random indices will be chosen. */
    data_management::NumericTablePtr correctionPairBatchIndices;
    /** Numeric table of size:
            - 1 x nIterations that contains values of the step-length sequence a(k), for k = 1, ..., nIterations, or
            - 1 x 1           that contains value of step length at each iteration a(1) = ... = a(nIterations) */
    data_management::NumericTablePtr stepLengthSequence;

    /**
    * Checks the correctness of the parameter
    *
     * \return Status of computations
    */
    virtual services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__INPUT"></a>
* \brief %Input class for LBFGS algorithm
*
* \snippet optimization_solver/lbfgs/lbfgs_types.h Input source code
*/
/* [Input source code] */
class DAAL_EXPORT Input : public optimization_solver::iterative_solver::Input
{
public:
    typedef optimization_solver::iterative_solver::Input super;
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);
    using super::set;
    using super::get;

    /**
    * Returns input NumericTable containing optional data
    * \param[in] id    Identifier of the input numeric table
    * \return          %Input numeric table that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalDataId id) const;

    /**
    * Sets optional input for the algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(OptionalDataId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks the correctness of the input
    * \param[in] par       Pointer to the structure of the algorithm parameters
    * \param[in] method    Computation method
    *
     * \return Status of computations
    */
    virtual services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};
/* [Input source code] */

/**
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LBFGS__RESULT"></a>
* \brief Results obtained with the compute() method of the LBFGS algorithm in the batch processing mode
*/
class DAAL_EXPORT Result : public optimization_solver::iterative_solver::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    typedef optimization_solver::iterative_solver::Result super;

    Result() {}
    using super::set;
    using super::get;

    /**
    * Allocates memory to store the results of the iterative solver algorithm
    * \param[in] input  Pointer to the input structure
    * \param[in] par    Pointer to the parameter structure
    * \param[in] method Computation method of the algorithm
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
    * Returns optional result of the algorithm
    * \param[in] id   Identifier of the optional result
    * \return         optional result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(OptionalDataId id) const;

    /**
    * Sets optional result of the algorithm
    * \param[in] id    Identifier of the optional result
    * \param[in] ptr   Pointer to the optional result
    */
    void set(OptionalDataId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks the result of the iterative solver algorithm
    * \param[in] input   %Input of algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method of the algorithm
    *
     * \return Status of computations
    */
    virtual services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                   int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;
};
typedef services::SharedPtr<Result> ResultPtr;
/* [Result source code] */

/** @} */
} // namespace interface2
using interface2::Parameter;
using interface2::Input;
using interface2::Result;
using interface2::ResultPtr;

} // namespace lbfgs
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
