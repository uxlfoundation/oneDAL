/* file: kernel_function.h */
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
//  Implementation of the kernel function interface.
//--
*/

#ifndef __KERNEL_FUNCTION_H__
#define __KERNEL_FUNCTION_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kernel_function/kernel_function_types.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace interface1
{
/**
 * @addtogroup kernel_function
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__KERNELIFACE"></a>
 * \brief Abstract class that specifies the interface of the algorithms
 *        for computing kernel functions in the batch processing mode
 */
class KernelIface : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::kernel_function::Input InputType;
    typedef algorithms::kernel_function::ParameterBase ParameterType;
    typedef algorithms::kernel_function::Result ResultType;

    KernelIface() { initialize(); }

    /**
     * Constructs an algorithm for computing kernel functions by copying input objects and parameters
     * of another algorithm for computing kernel functions
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    KernelIface(const KernelIface & /*other*/) { initialize(); }

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

    virtual ~KernelIface() {}

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

    /**
     * Returns a pointer to the newly allocated algorithm for computing kernel functions with a copy of input objects
     * and parameters of this algorithm for computing kernel functions
     * \return Pointer to the newly allocated algorithm
     */
    DAAL_FORCEINLINE services::SharedPtr<KernelIface> clone() const { return services::SharedPtr<KernelIface>(cloneImpl()); }

protected:
    void initialize() { _result = ResultPtr(new kernel_function::Result()); }
    virtual KernelIface * cloneImpl() const override = 0;
    ResultPtr _result;

private:
    KernelIface & operator=(const KernelIface &);
};
typedef services::SharedPtr<KernelIface> KernelIfacePtr;
/** @} */
} // namespace interface1
using interface1::KernelIface;
using interface1::KernelIfacePtr;

} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
