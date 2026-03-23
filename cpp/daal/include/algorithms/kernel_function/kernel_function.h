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
#include "services/daal_shared_ptr.h"

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
 *
 * \DAAL_DEPRECATED
 */
class DAAL_EXPORT KernelIface : public daal::algorithms::Analysis<batch>
{
public:
    virtual ~KernelIface() {}

    /**
     * Returns a pointer to the newly allocated algorithm for computing kernel functions with a copy of input objects
     * and parameters of this algorithm for computing kernel functions
     * \return Pointer to the newly allocated algorithm
     */
    DAAL_FORCEINLINE services::SharedPtr<KernelIface> clone() const { return services::SharedPtr<KernelIface>(cloneImpl()); }

protected:
    KernelIface() {}
    KernelIface(const KernelIface &) {}
    KernelIface * cloneImpl() const override = 0;

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
