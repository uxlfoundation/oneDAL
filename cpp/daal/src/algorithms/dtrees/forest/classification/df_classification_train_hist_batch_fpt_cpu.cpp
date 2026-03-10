/* file: df_classification_train_hist_batch_fpt_cpu.cpp */
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
//  Implementation of decision forest classification training functions for the hist method
//--
*/

#include "src/algorithms/dtrees/forest/classification/df_classification_train_container.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_train_dense_default_impl.i"
#include "src/algorithms/dtrees/forest/classification/df_classification_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{
template class BatchContainer<DAAL_FPTYPE, hist, DAAL_CPU>;
template class DAAL_EXPORT ClassificationTrainBatchKernel<DAAL_FPTYPE, hist, DAAL_CPU>;
}
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
