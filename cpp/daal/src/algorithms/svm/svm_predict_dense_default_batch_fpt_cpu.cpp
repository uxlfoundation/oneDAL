/* file: svm_predict_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of SVM Fast prediction algorithm.
//--
*/

#include "src/algorithms/svm/svm_predict_batch_container.h"
#include "src/algorithms/svm/svm_predict_kernel.h"
#include "src/algorithms/svm/svm_predict_impl.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace internal
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
template struct DAAL_EXPORT SVMPredictImpl<defaultDense, DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal
