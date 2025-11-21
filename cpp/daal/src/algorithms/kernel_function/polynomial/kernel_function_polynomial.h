/* file: kernel_function_polynomial.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef __KERNEL_FUNCTION_POLYNOMIAL_H__
#define __KERNEL_FUNCTION_POLYNOMIAL_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/kernel_function/polynomial/kernel_function_types_polynomial.h"
#include "algorithms/kernel_function/kernel_function.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace polynomial
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    BatchContainer(daal::services::Environment::env * daalEnv);

    ~BatchContainer();

    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public KernelIface
{
public:
    typedef KernelIface super;

    typedef algorithms::kernel_function::polynomial::internal::Input InputType;
    typedef algorithms::kernel_function::polynomial::internal::Parameter ParameterType;
    typedef typename super::ResultType ResultType;

    ParameterType parameter; /*!< Parameter of the kernel function*/
    InputType input;         /*!< %Input data structure */

    Batch();

    ~Batch();

    Batch(const Batch<algorithmFPType, method> & other);

    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    virtual InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    virtual ParameterBase * getParameter() DAAL_C11_OVERRIDE { return &parameter; }

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

} // namespace internal
} // namespace polynomial
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
