/* file: adagrad_batch.h */
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
//  Implementation of the interface for the adaptive gradient descent (Adagrad) algorithm
//  in the batch processing mode
//--
*/

#ifndef __ADAGRAD_BATCH_H__
#define __ADAGRAD_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "algorithms/optimization_solver/adagrad/adagrad_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace interface2
{
/**
 * @defgroup adagrad_batch Batch
 * @ingroup adagrad
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the adaptive gradient descent algorithm.
 *        This class is associated with daal::algorithms::optimization_solver::adagrad::BatchContainer class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Adaptive gradient descent algorithm, double or float
 * \tparam method           Adaptive gradient descent computation method, daal::algorithms::optimization_solver::adagrad::Method
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the Adagrad algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the Adagrad algorithm in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ADAGRAD__BATCH"></a>
 * \brief Computes Adaptive gradient descent in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SGD-ALGORITHM">Adaptive gradient descent algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Adaptive gradient descent algorithm,
 *                          double or float
 * \tparam method           Adaptive gradient descent computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for Adaptive gradient descent
 *      - \ref iterative_solver::InputId  Identifiers of input objects for Adaptive gradient descent
 *      - \ref iterative_solver::ResultId %Result identifiers for the Adaptive gradient descent
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public iterative_solver::Batch
{
public:
    typedef algorithms::optimization_solver::adagrad::Input InputType;
    typedef algorithms::optimization_solver::adagrad::Parameter ParameterType;
    typedef algorithms::optimization_solver::adagrad::Result ResultType;

    InputType input;         /*!< %Input data structure */
    ParameterType parameter; /*!< %Parameter data structure */

    /** Default constructor */
    Batch(sum_of_functions::BatchPtr objectiveFunction = sum_of_functions::BatchPtr());

    /**
     * Constructs a Adaptive gradient descent algorithm by copying input objects
     * of another Adaptive gradient descent algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Get input objects for the iterative solver algorithm
     * \return %Input objects for the iterative solver algorithm
     */
    virtual iterative_solver::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Get parameters of the iterative solver algorithm
     * \return Parameters of the iterative solver algorithm
     */
    virtual iterative_solver::Parameter * getParameter() DAAL_C11_OVERRIDE { return &parameter; }

    /**
     * Creates user-allocated memory to store results of the iterative solver algorithm
     *
     * \return Status of computations
     */
    virtual services::Status createResult() DAAL_C11_OVERRIDE
    {
        _result = iterative_solver::ResultPtr(new ResultType());
        _res    = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Adaptive gradient descent algorithm with a copy of input objects
     * of this Adaptive gradient descent algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    /**
    *  Creates the instance of the class
    *  \return     New instance of the class
    */
    static services::SharedPtr<Batch<algorithmFPType, method> > create();

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = static_cast<ResultType *>(_result.get())->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par                 = &parameter;
        _in                  = &input;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface2
using interface2::BatchContainer;
using interface2::Batch;

} // namespace adagrad
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
