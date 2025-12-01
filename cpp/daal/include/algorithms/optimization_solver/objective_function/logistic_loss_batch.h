/* file: logistic_loss_batch.h */
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
//  Implementation of the Logistic loss objective function in the batch
//  processing mode
//--
*/

#ifndef __LOGISTIC_LOSS_BATCH_H__
#define __LOGISTIC_LOSS_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_batch.h"
#include "algorithms/optimization_solver/objective_function/logistic_loss_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace logistic_loss
{
namespace interface2
{
/**
 * @defgroup logistic_loss_batch Batch
 * @ingroup logistic_loss
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the Logistic loss objective function.
 *        This class is associated with the Batch class and supports the method of computing
 *        the Logistic loss objective function in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Logistic loss objective function, double or float
 * \tparam method           the Logistic loss objective function computation method
 *
 * \DAAL_DEPRECATED
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for logistic loss objective function with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    DAAL_DEPRECATED BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of logistic loss objective function in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__LOGISTIC_LOSS__BATCH"></a>
 * \brief Computes the Logistic loss objective function in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-LOG-LOSS-ALGORITHM">The Logistic loss objective function algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Logistic loss objective function, double or float
 * \tparam method           The Logistic loss objective function computation method
 *
 * \par Enumerations
 *      - \ref Method Computation methods for the Logistic loss objective function
 *      - \ref InputId  Identifiers of input objects for the Logistic loss objective function
 *      - \ref objective_function::ResultId %Result identifiers for the Logistic loss objective function
 *
 * \par References
 *      - \ref objective_function::interface1::Result "Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public sum_of_functions::Batch
{
public:
    typedef sum_of_functions::Batch super;
    typedef algorithms::optimization_solver::logistic_loss::Input InputType;
    typedef algorithms::optimization_solver::logistic_loss::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    /**
     *  Main constructor
     */
    Batch(size_t numberOfTerms);

    virtual ~Batch() {}

    /**
     * Constructs an the Logistic loss objective function algorithm by copying input objects and parameters
     * of another the Logistic loss objective function algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> & other);

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated the Logistic loss objective function algorithm with a copy of input objects
     * of this the Logistic loss objective function algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

    /**
     * Allocates memory buffers needed for the computations
     *
     * \return Status of computations
     */
    services::Status allocate() { return allocateResult(); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType & parameter() { return *static_cast<ParameterType *>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType & parameter() const { return *static_cast<const ParameterType *>(_par); }

    /**
    *  Creates the instance of the class
    *  \param[in]  numberOfTerms  Constructor argument
    *  \return     New instance of the class
    */
    static services::SharedPtr<Batch<algorithmFPType, method> > create(size_t numberOfTerms);

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = sumOfFunctionsParameter;
    }

public:
    InputType input; /*!< %Input data structure */

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface2

using interface2::BatchContainer;
using interface2::Batch;

} // namespace logistic_loss
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
