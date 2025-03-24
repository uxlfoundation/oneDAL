/* file: adagrad_types.h */
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
//  Implementation of the Adaptive gradient descent algorithm types.
//--
*/

#ifndef __ADAGRAD_TYPES_H__
#define __ADAGRAD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "algorithms/engines/mt19937/mt19937.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
/**
 * @defgroup adagrad Adaptive Gradient Descent Algorithm
 * \copydoc daal::algorithms::optimization_solver::adagrad
 * @ingroup optimization_solver
 * @{
 */
/**
 * \brief Contains classes for computing the Adaptive gradient descent
 */
namespace adagrad
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__METHOD"></a>
 * Available methods for computing the Adaptive gradient descent
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__ADAGRAD_OPTIONALDATAID"></a>
* Available identifiers of optional input for the iterative solver
*/
enum OptionalDataId
{
    gradientSquareSum = iterative_solver::lastOptionalData + 1, /*!< Numeric table of size p x 1 with the values of G,
                                                                     where each value is an accumulated
                                                                     sum of squares of corresponding gradient's coordinate values. */
    lastOptionalData  = gradientSquareSum
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__PARAMETER"></a>
 * \brief %Parameter base class for the Adaptive gradient descent algorithm
 *
 * \snippet optimization_solver/adagrad/adagrad_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public optimization_solver::iterative_solver::Parameter
{
    /**
     * Constructs the parameter base class of the Adaptive gradient descent algorithm
     * \param[in] function_                 Objective function represented as sum of functions
     * \param[in] nIterations_              Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold_        Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices_             Numeric table that represents 32 bit integer indices of terms in the objective function.
     *                                      If no indices are provided, the implementation will generate random indices.
     * \param[in] batchSize_                Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                            in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                            This parameter is ignored if batchIndices is provided.
     * \param[in] learningRate_             Numeric table that contains value of the learning rate
     * \param[in] degenerateCasesThreshold_ Value needed to avoid degenerate cases in square root computing.
     * \param[in] seed_                     Seed for random generation of 32 bit integer indices of terms in the objective function. \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(const sum_of_functions::BatchPtr & function_, size_t nIterations_ = 100, double accuracyThreshold_ = 1.0e-05,
              data_management::NumericTablePtr batchIndices_ = data_management::NumericTablePtr(), const size_t batchSize_ = 128,
              data_management::NumericTablePtr learningRate_ =
                  data_management::HomogenNumericTable<double>::create(1, 1, data_management::NumericTableIface::doAllocate, 0.01),
              double degenerateCasesThreshold_ = 1.0e-08, size_t seed_ = 777);

    virtual ~Parameter() {}

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const DAAL_C11_OVERRIDE;

    data_management::NumericTablePtr batchIndices; /*!< Numeric table that represents 32 bit integer indices of terms
                                                                   in the objective function. If no indices are provided,
                                                                   the implementation will generate random indices. */
    data_management::NumericTablePtr learningRate; /*!< Numeric table that contains value of the learning rate */
    double degenerateCasesThreshold;               /*!< Value needed to avoid degenerate cases in square root computing. */
    size_t seed;                                   /*!< Seed for random generation of 32 bit integer indices of terms
                                                                   in the objective function. \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine;                     /*!< Engine for random generation of 32 bit integer indices of terms
                                                                   in the objective function. */
};
/* [Parameter source code] */

/**
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__INPUT"></a>
* \brief %Input class for the Adaptive gradient descent algorithm
*
* \snippet optimization_solver/adagrad/adagrad_types.h Input source code
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
    * Returns input NumericTable containing accumulated sum of gradients
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
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__RESULT"></a>
* \brief Results obtained with the compute() method of the adagrad algorithm in the batch processing mode
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

} // namespace adagrad
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
