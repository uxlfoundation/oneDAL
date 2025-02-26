/* file: pivoted_qr_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of pivoted_qr calculation algorithm container.
//--
*/

#include "src/algorithms/pivoted_qr/pivoted_qr_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(pivoted_qr::BatchContainer, batch, DAAL_FPTYPE, pivoted_qr::defaultDense)
namespace pivoted_qr
{
namespace interface1
{
template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch()
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(const Batch<algorithmFPType, method> & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

template class Batch<DAAL_FPTYPE, pivoted_qr::defaultDense>;
} //namespace interface1
} //namespace pivoted_qr
} //namespace algorithms
} //namespace daal
