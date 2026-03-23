/* file: kernel_function_iface_impl.h */
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
//  Internal implementation of the kernel function interface with full details.
//--
*/

#ifndef __KERNEL_FUNCTION_IFACE_IMPL_H__
#define __KERNEL_FUNCTION_IFACE_IMPL_H__

#include "algorithms/kernel_function/kernel_function.h"
#include "src/algorithms/kernel_function/kernel_function_types.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace internal
{
/**
 * \brief Internal base class with full implementation details for kernel function algorithms.
 *        This class is used internally in src/ and provides the complete interface
 *        that was previously in the public KernelIface.
 */
class KernelIfaceImpl : public kernel_function::KernelIface
{
public:
    typedef algorithms::kernel_function::Input InputType;
    typedef algorithms::kernel_function::ParameterBase ParameterType;
    typedef algorithms::kernel_function::Result ResultType;

    KernelIfaceImpl() { initialize(); }

    /**
     * Constructs an algorithm for computing kernel functions by copying input objects and parameters
     * of another algorithm for computing kernel functions
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    KernelIfaceImpl(const KernelIfaceImpl & /*other*/) { initialize(); }

    /**
     * Get input objects for the kernel function algorithm
     * \return %Input objects for the kernel function algorithm
     */
    virtual Input * getInput() = 0;

    /**
     * Get parameters of the kernel function algorithm
     * \return Parameters of the kernel function algorithm
     */
    virtual ParameterBase * getParameter() = 0;

    virtual ~KernelIfaceImpl() {}

    /**
     * Returns the structure that contains computed results of the kernel function algorithm
     * \returns the Structure that contains computed results of the kernel function algorithm
     */
    ResultPtr getResult() { return _result; }

    /**
     * Registers user-allocated memory to store results of the kernel function algorithm
     * \param[in] res  Structure to store the results
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

protected:
    void initialize() { _result = ResultPtr(new kernel_function::Result()); }
    KernelIfaceImpl * cloneImpl() const override = 0;
    ResultPtr _result;

private:
    KernelIfaceImpl & operator=(const KernelIfaceImpl &);
};

} // namespace internal
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
