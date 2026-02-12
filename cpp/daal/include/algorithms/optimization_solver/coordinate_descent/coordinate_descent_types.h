/* file: coordinate_descent_types.h */
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
//  Implementation of the Coordinate descent algorithm types.
//--
*/

#ifndef __COORDINATE_DESCENT_TYPES_H__
#define __COORDINATE_DESCENT_TYPES_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "algorithms/engines/mt19937/mt19937.h"
#include "algorithms/optimization_solver/objective_function/logistic_loss_batch.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * @defgroup coordinate_descent Coordinate Descent Algorithm
 * \copydoc daal::algorithms::optimization_solver::coordinate_descent
 * @ingroup optimization_solver
 * @{
 */
/**
 * \brief Contains classes for computing the Coordinate descent
 */
namespace coordinate_descent
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__COORDINATE_DESCENT__METHOD"></a>
 * Available methods for computing the Coordinate descent
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__COORDINATE_DESCENT__METHOD"></a>
 * Available coordinate selection strategy for computing the Coordinate descent
 */
enum SelectionStrategy
{
    cyclic, /*!< Cyclic selection of coordinate to be optomized */
    random  /*!< Random selection of coordinate to be optomized */
};
/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__COORDINATE_DESCENT__PARAMETER"></a>
 * \brief %Parameter base class for the Coordinate descent algorithm
 *
 * \snippet optimization_solver/coordinate_descent/coordinate_descent_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public optimization_solver::iterative_solver::Parameter
{
    /**
     * Constructs the parameter base class of the Coordinate descent algorithm
     * \param[in] function                 Objective function represented as sum of functions
     * \param[in] nIterations              Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold        Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     *                                      If no indices are provided, the implementation will generate random indices.
     * \param[in] seed                     Seed for random generation of 32 bit integer indices of terms in the objective function. \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(const sum_of_functions::BatchPtr & function, size_t nIterations = 100, double accuracyThreshold = 1.0e-05, size_t seed = 777);

    virtual ~Parameter() {}

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const DAAL_C11_OVERRIDE;

    size_t seed;               /*!< Seed for random generation of 32 bit integer indices of terms
                                                      in the objective function. \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine; /*!< Engine for random generation of 32 bit integer indices of terms
                                                                   in the objective function. */
    SelectionStrategy selection;
    bool positive;
    bool skipTheFirstComponents;
};
/* [Parameter source code] */

/**
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__COORDINATE_DESCENT__INPUT"></a>
* \brief %Input class for the Coordinate descent algorithm
*
* \snippet optimization_solver/coordinate_descent/coordinate_descent_types.h Input source code
*/
/* [Input source code] */
class DAAL_EXPORT Input : public optimization_solver::iterative_solver::Input
{
private:
    typedef optimization_solver::iterative_solver::Input super;

public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);

    using super::set;
    using super::get;

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
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__COORDINATE_DESCENT__RESULT"></a>
* \brief Results obtained with the compute() method of the coordinate_descent algorithm in the batch processing mode
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
    *
     * \return Status of computations
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

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
} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
