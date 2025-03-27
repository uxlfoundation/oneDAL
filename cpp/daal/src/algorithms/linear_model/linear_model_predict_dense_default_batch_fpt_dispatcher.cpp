/* file: linear_model_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of linear regression algorithm container -- a class
//  that contains fast linear regression prediction kernels
//  for supported architectures.
//--
*/

#include "src/algorithms/linear_model/linear_model_predict_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(linear_model::prediction::BatchContainer, batch, DAAL_FPTYPE, linear_model::prediction::defaultDense)
namespace linear_model
{
namespace prediction
{
namespace interface1
{
template <>
DAAL_EXPORT void Batch<DAAL_FPTYPE, linear_model::prediction::defaultDense>::initialize()
{
    this->_ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, DAAL_FPTYPE, defaultDense)(&(this->_env));
    this->_par = NULL;
}
} // namespace interface1
} // namespace prediction
} // namespace linear_model
} // namespace algorithms
} // namespace daal
