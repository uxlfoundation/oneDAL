/* file: multiclassclassifier_train_kernel.h */
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
//  Declaration of template function that trains Multi-class slassifier model.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_KERNEL_H__
#define __MULTICLASSCLASSIFIER_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/algorithm.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_train_types.h"
#include "src/algorithms/svm/svm_model_impl.h"

#include "src/services/service_defines.h"

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
using namespace daal::data_management;
using namespace daal::internal;

struct KernelParameter
{
    size_t nClasses;                                                           /*!< Number of classes */
    services::SharedPtr<algorithms::classifier::training::Batch> training;     /*!< Two-class classifier training stage */
    services::SharedPtr<algorithms::classifier::prediction::Batch> prediction; /*!< Two-class classifier prediction stage */
    size_t maxIterations;                                                      /*!< Maximum number of iterations */
    double accuracyThreshold;                                                  /*!< Convergence threshold */
};

template <Method method, typename AlgorithmFPtype, CpuType cpu>
struct MultiClassClassifierTrainKernel : public Kernel
{
    services::Status compute(const NumericTable * a0, const NumericTable * a1, const NumericTable * a2, daal::algorithms::Model * r,
                             svm::internal::ModelImpl * svmModel, const KernelParameter & par);
};

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
