/* file: logistic_loss_types.h */
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
//  Implementation of logistic loss objective function
//--
*/

#ifndef __LOGISTIC_LOSS_TYPES_H__
#define __LOGISTIC_LOSS_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the Logistic loss objective function
 */
namespace optimization_solver
{
/**
 * @defgroup logistic_loss Logistic loss Algorithm
 * \copydoc daal::algorithms::optimization_solver::logistic_loss
 * @ingroup objective_function
 * @{
 */
/**
* \brief Contains classes for computing the Logistic loss objective function
*/
namespace logistic_loss
{
/**
  * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__INPUTID"></a>
  * Available identifiers of input objects of the Logistic loss objective function
  */
enum InputId
{
    argument = (int)sum_of_functions::argument, /*!< Numeric table of size 1 x p with input argument of the objective function */
    data,                                       /*!< Numeric table of size n x p with data */
    dependentVariables,                         /*!< Numeric table of size n x 1 with dependent variables */
    lastInputId = dependentVariables
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__METHOD"></a>
 * Available methods for computing results of Logistic loss objective function
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
 * \brief Contains version 2.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__PARAMETER"></a>
 * \brief %Parameter for Logistic loss objective function
 *
 * \snippet optimization_solver/objective_function/logistic_loss_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public sum_of_functions::Parameter
{
    /**
     * Constructs the parameter of Logistic loss objective function
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                                   a batch of indices used to compute the function results, e.g.,
                                   value of the sum of the functions. If no indices are provided,
                                   all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t numberOfTerms, data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(),
              const DAAL_UINT64 resultsToCompute = objective_function::gradient);

    /**
     * Constructs a Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source object to initialize from
     */
    Parameter(const Parameter & other);

    /**
     * Assigns a Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source object to copy from
     */
    Parameter & operator=(const Parameter & other);

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;

    virtual ~Parameter() {}

    bool interceptFlag; /*!< Whether the intercept needs to be computed. Default is true */
    float penaltyL1;    /*!< L1 regularization coefficient. Default is 0 (not applied) */
    float penaltyL2;    /*!< L2 regularization coefficient. Default is 0 (not applied) */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__MSE__INPUT"></a>
 * \brief %Input objects for the Logistic loss objective function
 */
class DAAL_EXPORT Input : public sum_of_functions::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other);

    /** Assignment operator */
    Input & operator=(const Input & other);

    /** Destructor */
    virtual ~Input() {}

    /**
     * Sets one input object for Logistic loss objective function
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Returns the input numeric table for Logistic loss objective function
     * \param[in] id    Identifier of the input numeric table
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface2
using interface2::Parameter;
using interface2::Input;

} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
