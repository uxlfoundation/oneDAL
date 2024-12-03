/* file: mrg32k3a_dense_default_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright contributors to the oneDAL project
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
//  Implementation of mrg32k3a calculation algorithm dispatcher.
//--

#include "src/algorithms/engines/mrg32k3a/mrg32k3a_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(engines::mrg32k3a::BatchContainer, batch, DAAL_FPTYPE, engines::mrg32k3a::defaultDense)
} // namespace algorithms
} // namespace daal
