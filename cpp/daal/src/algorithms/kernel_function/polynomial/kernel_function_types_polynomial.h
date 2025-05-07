/* file: kernel_function_types_polynomial.h */
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

#ifndef __KERNEL_FUNCTION_TYPES_POLYNOMIAL_H__
#define __KERNEL_FUNCTION_TYPES_POLYNOMIAL_H__

#include "algorithms/kernel_function/kernel_function_types.h"
#include "src/algorithms/kernel_function/kernel_function_dense_base.h"

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
enum Method
{
    defaultDense = 0, /*!< Default method for computing polynomial kernel functions */
    fastCSR      = 1  /*!< Fast: performance-oriented method. Works with Compressed Sparse Rows (CSR) numeric tables */
};

struct DAAL_EXPORT Parameter : public ParameterBase
{
    Parameter(double scale = 1.0, double shift = 0.0, size_t degree = 3);
    double scale;  /*!< Polynomial kernel coefficient k in the (k(X,Y) + b)^d model */
    double shift;  /*!< Polynomial kernel coefficient b in the (k(X,Y) + b)^d model */
    size_t degree; /*!< Polynomial kernel coefficient d in the (k(X,Y) + b)^d model */
    KernelType kernelType = KernelType::polynomial;
};

class DAAL_EXPORT Input : public kernel_function::Input
{
public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);
    ~Input() override;

    services::Status check(const daal::algorithms::Parameter * par, int method) const override;
};

} // namespace internal
} // namespace polynomial
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
