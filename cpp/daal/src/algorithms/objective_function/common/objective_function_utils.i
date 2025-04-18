/* file: objective_function_utils.i */
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
//  Implementation of objective function utilities
//--
*/
#include "src/externals/service_math.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{
namespace internal
{
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
services::Status getXY(NumericTable * dataNT, NumericTable * dependentVariablesNT, const NumericTable * indNT, algorithmFPType * aX,
                       algorithmFPType * aY, size_t nRows, size_t n, size_t p)
{
    DAAL_PROFILER_TASK(getXY);
    DAAL_ASSERT(indNT != nullptr);
    DAAL_ASSERT(dataNT != nullptr);
    DAAL_ASSERT(dependentVariablesNT != nullptr);
    DAAL_ASSERT(aX != nullptr);
    DAAL_ASSERT(aY != nullptr);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, n, p);

    ReadRows<int, cpu> rInd(*const_cast<NumericTable *>(indNT), 0, n);
    DAAL_CHECK_BLOCK_STATUS(rInd);
    const int * ind = rInd.get();

    ReadRows<algorithmFPType, cpu> xr(*dataNT);
    DAAL_CHECK_BLOCK_STATUS(xr);
    ReadRows<algorithmFPType, cpu> yr(*dependentVariablesNT, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(yr);
    for (size_t i = 0; i < n; ++i)
    {
        xr.next(ind[i], 1);
        services::internal::tmemcpy<algorithmFPType, cpu>(aX + i * p, xr.get(), p);
        aY[i] = yr.get()[ind[i]];
    }
    return services::Status();
}

} // namespace internal

} // namespace objective_function

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
