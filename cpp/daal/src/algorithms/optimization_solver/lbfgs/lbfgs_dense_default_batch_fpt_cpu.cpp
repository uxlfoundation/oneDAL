/* file: lbfgs_dense_default_batch_fpt_cpu.cpp */
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

//++
//  Instantiation of LBFGS computation functions for specific CPUs.
//--

#include "src/algorithms/optimization_solver/lbfgs/lbfgs_batch_container.h"
#include "src/algorithms/optimization_solver/lbfgs/lbfgs_dense_default_kernel.h"
#include "src/algorithms/optimization_solver/lbfgs/lbfgs_dense_default_impl.i"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace internal
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
template class LBFGSKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
