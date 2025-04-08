/* file: tsne_gradient_descent_fpt.cpp */
/*******************************************************************************
* Copyright 2022 Intel Corporation
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
//  Instantiation of tSNE interface function
//--
*/

#include "algorithms/tsne/tsne_gradient_descent.h"
#include "tsne_gradient_descent_kernel.h"

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename algorithmIdxType, typename algorithmFPType>
DAAL_EXPORT void tsneGradientDescent(const NumericTablePtr initTable, const CSRNumericTablePtr pTable, const NumericTablePtr sizeIterTable,
                                     const NumericTablePtr paramTable, const NumericTablePtr resultTable)
{
#define DAAL_TSNE_GRADIENT_DESCENT(cpuId, ...) tsneGradientDescentImpl<algorithmIdxType, algorithmFPType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_TSNE_GRADIENT_DESCENT, initTable, pTable, sizeIterTable, paramTable, resultTable);

#undef DAAL_TSNE_GRADIENT_DESCENT
}

template DAAL_EXPORT void tsneGradientDescent<int, DAAL_FPTYPE>(const NumericTablePtr initTable, const CSRNumericTablePtr pTable,
                                                                const NumericTablePtr sizeIterTable, const NumericTablePtr paramTable,
                                                                const NumericTablePtr resultTable);

} // namespace internal
} // namespace algorithms
} // namespace daal
