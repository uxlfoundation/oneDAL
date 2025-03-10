/* file: kernel_function_linear.h */
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
//  Implementation of the interface for the linear kernel function algorithm
//--
*/

#ifndef __KERNEL_FUNCTION_LINEAR_H__
#define __KERNEL_FUNCTION_LINEAR_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kernel_function/kernel_function_types_linear.h"
#include "algorithms/kernel_function/kernel_function.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace linear
{
namespace interface1
{
/**
 * @defgroup kernel_function_linear_batch Batch
 * @ingroup kernel_function_linear
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__LINEAR__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the linear kernel function algorithm.
 *        This class is associated with the Batch class
 *        and supports the method for computing linear kernel functions in the %batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of kernel functions, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 */

template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the linear kernel function algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the linear kernel function algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__LINEAR__BATCH"></a>
 * \brief Computes a linear kernel function in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-KERNEL_FUNCTION_LINEAR-ALGORITHM">Kernel function algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations  of kernel functions, double or float
 * \tparam method           Computation method of the algorithm, \ref Method
 *
 * \par Enumerations
 *      - \ref Method   Methods for computing  kernel functions
 *      - \ref InputId  Identifiers of input objects for the kernel function algorithm
 *      - \ref ResultId Identifiers of results of the kernel function algorithm
 *
 * \par References
 *      - \ref interface1::Result "Result" class
 */
template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public KernelIface
{
public:
    typedef KernelIface super;

    typedef algorithms::kernel_function::linear::Input InputType;
    typedef algorithms::kernel_function::linear::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    ParameterType parameter; /*!< Parameter of the kernel function*/
    InputType input;         /*!< %Input data structure */

    /** Default constructor */
    Batch();

    /**
     * Constructs linear kernel function algorithm by copying input objects and parameters
     * of another linear kernel function algorithm
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
     * Get input objects for the kernel function algorithm
     * \return %Input objects for the kernel function algorithm
     */
    virtual InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Get parameters of the kernel function algorithm
     * \return Parameters of the kernel function algorithm
     */
    virtual ParameterBase * getParameter() DAAL_C11_OVERRIDE { return &parameter; }

    /**
     * Returns a pointer to the newly allocated linear kernel function algorithm with a copy of input objects
     * and parameters of this linear kernel function algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in                  = &input;
        _par                 = &parameter;
    }

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE { return new Batch<algorithmFPType, method>(*this); }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int)method);
        _res               = _result.get();
        return s;
    }

private:
    Batch & operator=(const Batch &);
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
