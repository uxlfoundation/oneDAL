/* file: tsne_gradient_descent_fpt_cpu.cpp */
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
//  Instantiation of CPU-specific tSNE kernel implementations
//--
*/

#include "tsne_gradient_descent_kernel.h"
#include "tsne_gradient_descent_impl.i"

#if defined(DAAL_INTEL_CPP_COMPILER)
    #if (__CPUID__(DAAL_CPU) == __avx512__)

        #include <immintrin.h>
        #include "tsne_gradient_descent_avx512_impl.i"

    #endif // __CPUID__(DAAL_CPU) == __avx512__
#endif     // DAAL_INTEL_CPP_COMPILER

namespace daal
{
namespace algorithms
{
namespace internal
{
template DAAL_EXPORT services::Status tsneGradientDescentImpl<int, DAAL_FPTYPE, DAAL_CPU>(const NumericTablePtr initTable,
                                                                                          const CSRNumericTablePtr pTable,
                                                                                          const NumericTablePtr sizeIterTable,
                                                                                          const NumericTablePtr paramTable,
                                                                                          const NumericTablePtr resultTable);

} // namespace internal
} // namespace algorithms
} // namespace daal
