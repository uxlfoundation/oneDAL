/* file: sgd_types.h */
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
//  Implementation of the Stochastic gradient descent algorithm types.
//--
*/

#ifndef __SGD_TYPES_H__
#define __SGD_TYPES_H__

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
 * @defgroup sgd Stochastic Gradient Descent Algorithm
 * \copydoc daal::algorithms::optimization_solver::sgd
 * @ingroup optimization_solver
 * @{
 */
/**
 * \brief Contains classes for computing the Stochastic gradient descent
 */
namespace sgd
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__METHOD"></a>
 * Available methods for computing the Stochastic gradient descent
 */
enum Method
{
    defaultDense = 0, /*!< Default: Required gradient is computed using only one term of objective function */
    miniBatch    = 1, /*!< Required gradient is computed using batchSize terms of objective function  */
    momentum     = 2  /*!< Required gradient is computed using batchSize terms of objective function, perform momentum update rule  */
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__SGD__OPTIONALDATAID"></a>
* Available identifiers of optional input for the iterative solver
*/
enum OptionalDataId
{
    pastUpdateVector =
        iterative_solver::lastOptionalData + 1, /*!< NumericTable of size p x 1 with vector update from past iteration. Applied for momentum method */
    pastWorkValue =
        pastUpdateVector + 1, /*!< NumericTable of size p x 1 with work vector value from past main iteration. Applied for minibatch method */
    lastOptionalData = pastWorkValue
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__BASEPARAMETER"></a>
 * \brief %BaseParameter base class for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h BaseParameter source code
 */
/* [BaseParameter source code] */
struct DAAL_EXPORT BaseParameter : public optimization_solver::iterative_solver::Parameter
{
    /**
     * Constructs the parameter base class of the Stochastic gradient descent algorithm
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function.
     *                                 If no indices are provided, the implementation will generate random indices.
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] batchSize            Batch size
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function. \DAAL_DEPRECATED_USE{ engine }
     */
    BaseParameter(const sum_of_functions::BatchPtr & function, size_t nIterations = 100, double accuracyThreshold = 1.0e-05,
                  data_management::NumericTablePtr batchIndices         = data_management::NumericTablePtr(),
                  data_management::NumericTablePtr learningRateSequence = data_management::NumericTablePtr(
                      new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
                  size_t batchSize = 1, size_t seed = 777);

    virtual ~BaseParameter() {}

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;

    data_management::NumericTablePtr batchIndices;         /*!< Numeric table that represents 32 bit integer indices of terms
                                                                in the objective function. If no indices are provided,
                                                                the implementation will generate random indices. */
    data_management::NumericTablePtr learningRateSequence; /*!< Numeric table that contains values of the learning rate sequence */
    size_t seed;                                           /*!< Seed for random generation of 32 bit integer indices of terms
                                                                  in the objective function. \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine;                             /*!< Engine for random generation of 32 bit integer indices of terms
                                                                  in the objective function. */
};
/* [BaseParameter source code] */

template <Method method>
struct Parameter : public BaseParameter
{};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETER_DEFAULTDENSE"></a>
 * \brief %Parameter for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h ParameterDefaultDense source code
 */
/* [ParameterDefaultDense source code] */
template <>
struct DAAL_EXPORT Parameter<defaultDense> : public BaseParameter
{
    /**
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices are
                                       provided, the implementation will generate random indices.
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function. \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(const sum_of_functions::BatchPtr & function, size_t nIterations = 100, double accuracyThreshold = 1.0e-05,
              data_management::NumericTablePtr batchIndices         = data_management::NumericTablePtr(),
              data_management::NumericTablePtr learningRateSequence = data_management::NumericTablePtr(
                  new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
              size_t seed = 777);

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;

    virtual ~Parameter() {}
};
/* [ParameterDefaultDense source code] */

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETER_MINIBATCH"></a>
 * \brief %Parameter for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h ParameterMiniBatch source code
 */
/* [ParameterMiniBatch source code] */
template <>
struct DAAL_EXPORT Parameter<miniBatch> : public BaseParameter
{
    /**
     * Constructs the parameter class of the Stochastic gradient descent algorithm
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices
                                       are provided, the implementation will generate random indices.
     * \param[in] batchSize            Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                       in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                       This parameter is ignored if batchIndices is provided.
     * \param[in] conservativeSequence Numeric table of values of the conservative coefficient sequence
     * \param[in] innerNIterations     Number of inner iterations
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function. \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(const sum_of_functions::BatchPtr & function, size_t nIterations = 100, double accuracyThreshold = 1.0e-05,
              data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(), size_t batchSize = 128,
              data_management::NumericTablePtr conservativeSequence = data_management::NumericTablePtr(
                  new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
              size_t innerNIterations                               = 5,
              data_management::NumericTablePtr learningRateSequence = data_management::NumericTablePtr(
                  new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
              size_t seed = 777);

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;

    virtual ~Parameter() {}

    data_management::NumericTablePtr conservativeSequence; /*!< Numeric table of values of the conservative coefficient sequence */
    size_t innerNIterations;
};
/* [ParameterMiniBatch source code] */
/** @} */

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETER_MINIBATCH"></a>
 * \brief %Parameter for the Stochastic gradient descent algorithm
 *
 * \snippet optimization_solver/sgd/sgd_types.h ParameterMomentum source code
 */
/* [ParameterMomentum source code] */
template <>
struct DAAL_EXPORT Parameter<momentum> : public BaseParameter
{
    /**
     * Constructs the parameter class of the Stochastic gradient descent algorithm
     * \param[in] function             Objective function represented as sum of functions
     * \param[in] momentum             The momentum value
     * \param[in] nIterations          Maximal number of iterations of the algorithm
     * \param[in] accuracyThreshold    Accuracy of the algorithm. The algorithm terminates when this accuracy is achieved
     * \param[in] batchIndices         Numeric table that represents 32 bit integer indices of terms in the objective function. If no indices
                                       are provided, the implementation will generate random indices.
     * \param[in] batchSize            Number of batch indices to compute the stochastic gradient. If batchSize is equal to the number of terms
                                       in objective function then no random sampling is performed, and all terms are used to calculate the gradient.
                                       This parameter is ignored if batchIndices is provided.
     * \param[in] learningRateSequence Numeric table that contains values of the learning rate sequence
     * \param[in] seed                 Seed for random generation of 32 bit integer indices of terms in the objective function. \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(const sum_of_functions::BatchPtr & function, double momentum = 0.9, size_t nIterations = 100, double accuracyThreshold = 1.0e-05,
              data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(), size_t batchSize = 128,
              data_management::NumericTablePtr learningRateSequence = data_management::NumericTablePtr(
                  new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate, 1.0)),
              size_t seed = 777);

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;

    virtual ~Parameter() {}

    double momentum; /*!< Momentum value */
};
/* [ParameterMomentum source code] */
/** @} */

/**
* <a name="DAAL-STRUCT-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__INPUT"></a>
* \brief %Input for the Stochastic gradient descent algorithm
*
* \snippet optimization_solver/sgd/sgd_types.h Input source code
*/
/* [Input source code] */
class DAAL_EXPORT Input : public optimization_solver::iterative_solver::Input
{
public:
    typedef optimization_solver::iterative_solver::Input super;
    Input();
    Input(const Input & other);
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
/** @} */

/**
* <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__RESULT"></a>
* \brief Results obtained with the compute() method of the sgd algorithm in the batch processing mode
*/
/* [Result source code] */
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
using interface2::BaseParameter;
using interface2::Parameter;
using interface2::Input;
using interface2::Result;
using interface2::ResultPtr;

} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
