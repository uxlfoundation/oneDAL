/* file: implicit_als_train_init_csr_default_distr_step2_fpt_cpu.cpp */
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
//  Implementation of implicit ALS initialization functions for default CSR method.
//--
*/

#include "src/algorithms/implicit_als/implicit_als_train_init_kernel.h"
#include "src/algorithms/implicit_als/implicit_als_train_init_csr_default_distr_impl.i"
#include "src/algorithms/implicit_als/implicit_als_train_init_container.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{

namespace internal
{
template class DistributedContainer<step2Local, DAAL_FPTYPE, fastCSR, DAAL_CPU>;

template class ImplicitALSInitDistrStep2Kernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
