/* file: svm_train_boser_batch_fpt_dispatcher.cpp */
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
//  Implementation of SVM training algorithm container.
//--
*/

#include "src/algorithms/svm/svm_train_batch_container.h"
#include "src/algorithms/svm/svm_train_internal.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(svm::training::internal::BatchContainer, batch, DAAL_FPTYPE, svm::training::boser)
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(svm::training::internal::NuBatchContainer, batch, DAAL_FPTYPE, svm::training::boser)
namespace svm
{
namespace training
{
namespace interface2
{
using BatchType = Batch<DAAL_FPTYPE, svm::training::boser>;

template <>
void BatchType::initialize()
{
    _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, internal::BatchContainer, DAAL_FPTYPE, svm::training::boser)(&_env);
    _in  = &input;
    _par = &parameter;
    _result.reset(new ResultType());
}
template <>
DAAL_EXPORT BatchType::Batch()
{
    initialize();
}

template <>
DAAL_EXPORT BatchType::Batch(size_t nClasses)
{
    parameter.nClasses = nClasses;
    initialize();
}

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : classifier::training::Batch(other), parameter(other.parameter), input(other.input)
{
    initialize();
}

} // namespace interface2

// This is different from all other algorithms due to PR #1779, for a NuSVC Multiclass fix
namespace internal
{
using BatchType = Batch<DAAL_FPTYPE, svm::training::boser>;

template <>
DAAL_EXPORT BatchType::Batch()
{
    initialize();
}

template <>
DAAL_EXPORT BatchType::Batch(size_t nClasses)
{
    parameter.nClasses = nClasses;
    initialize();
}

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : classifier::training::Batch(other), parameter(other.parameter), input(other.input)
{
    initialize();
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
