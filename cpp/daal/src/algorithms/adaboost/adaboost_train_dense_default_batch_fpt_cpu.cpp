/* file: adaboost_train_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of Freund method for Ada Boost training algorithm.
//--
*/

#include "src/algorithms/adaboost/adaboost_train_batch_container.h"
#include "src/algorithms/adaboost/adaboost_train_kernel.h"
#include "src/algorithms/adaboost/adaboost_train_impl.i"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace training
{
namespace interface2
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
template class BatchContainer<DAAL_FPTYPE, sammeR, DAAL_CPU>;
} // namespace interface2
namespace internal
{
template class DAAL_EXPORT AdaBoostTrainKernel<defaultDense, DAAL_FPTYPE, DAAL_CPU>;
template class DAAL_EXPORT AdaBoostTrainKernel<sammeR, DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace training
} // namespace adaboost
} // namespace algorithms
} // namespace daal
