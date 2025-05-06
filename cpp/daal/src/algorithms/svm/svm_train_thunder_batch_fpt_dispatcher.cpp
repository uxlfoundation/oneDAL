/* file: svm_train_thunder_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(svm::training::BatchContainer, batch, DAAL_FPTYPE, svm::training::thunder)
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(svm::training::internal::BatchContainer, batch, DAAL_FPTYPE, svm::training::thunder)
namespace svm
{
namespace training
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, svm::training::thunder>::Batch()
{
    initialize();
}

template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, svm::training::thunder>::Batch(size_t nClasses)
{
    parameter.nClasses = nClasses;
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, svm::training::thunder>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : classifier::training::Batch(other), parameter(other.parameter), input(other.input)
{
    initialize();
}

} // namespace interface2

// This is different from all other algorithms due to PR #1779, for a NuSVC Multiclass fix
namespace internal
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, svm::training::thunder>::Batch()
{
    initialize();
}

template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, svm::training::thunder>::Batch(size_t nClasses)
{
    parameter.nClasses = nClasses;
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, svm::training::thunder>;

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
