/* file: multiclassclassifier_train_oneagainstone_batch_fpt_cpu.cpp */
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
//  Implementation of One-Against-One method for Multi-class classifier
//  training algorithm.
//--
*/

#include "src/algorithms/multiclassclassifier/multiclassclassifier_train_batch_container.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_train_oneagainstone_kernel.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_train_oneagainstone_impl.i"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace internal
{
template class BatchContainer<DAAL_FPTYPE, oneAgainstOne, DAAL_CPU>;
template class DAAL_EXPORT MultiClassClassifierTrainKernel<oneAgainstOne, DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
