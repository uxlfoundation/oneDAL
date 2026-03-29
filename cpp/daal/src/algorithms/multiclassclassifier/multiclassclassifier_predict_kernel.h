/* file: multiclassclassifier_predict_kernel.h */
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
//  Declaration of template function that computes prediction results using
//  Multi-class classifier model.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_PREDICT_KERNEL_H__
#define __MULTICLASSCLASSIFIER_PREDICT_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/algorithm.h"
#include "algorithms/multi_class_classifier/multi_class_classifier_predict_types.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_svm_model.h"
#include "src/services/service_defines.h"
#include "src/services/service_arrays.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;
using namespace multi_class_classifier::internal;

template <prediction::Method pmethod, training::Method tmethod, typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierPredictKernel : public Kernel
{
    services::Status compute(const NumericTable * a, const daal::algorithms::Model * m, SvmModel * svmModel, NumericTable * pred, NumericTable * df,
                             const daal::algorithms::Parameter * par);
};

template <typename algorithmFPType, CpuType cpu>
size_t getMultiClassClassifierPredictBlockSize()
{
    return 128;
}

template <typename algorithmFPType, CpuType cpu>
services::Status getClassIndices(size_t nClasses, bool isSvmModel, size_t * classIndicesData)
{
    const size_t nModels = (nClasses * (nClasses - 1)) >> 1;
    if (isSvmModel)
    {
        for (size_t iClass = 0, imodel = 0; iClass < nClasses; ++iClass)
        {
            for (size_t jClass = iClass + 1; jClass < nClasses; ++jClass, ++imodel)
            {
                classIndicesData[imodel]           = iClass;
                classIndicesData[imodel + nModels] = jClass;
            }
        }
    }
    else
    {
        for (size_t iClass = 1, imodel = 0; iClass < nClasses; ++iClass)
        {
            for (size_t jClass = 0; jClass < iClass; ++jClass, ++imodel)
            {
                classIndicesData[imodel]           = iClass;
                classIndicesData[imodel + nModels] = jClass;
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status getNonEmptyClassMap(size_t & nClasses, const Model * model, const size_t * classIndicesData, size_t * nonEmptyClassMap)
{
    TArray<bool, cpu> nonEmptyClassBuffer(nClasses);
    DAAL_CHECK_MALLOC(nonEmptyClassBuffer.get());
    bool * nonEmptyClass = (bool *)nonEmptyClassBuffer.get();
    for (size_t i = 0; i < nClasses; i++) nonEmptyClass[i] = false;

    const size_t nModels = (nClasses * (nClasses - 1)) >> 1;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nModels, 2);
    TArray<size_t, cpu> classIndices(nModels * 2);
    DAAL_CHECK_MALLOC(classIndices.get());

    for (size_t imodel = 0; imodel < nModels; ++imodel)
    {
        const size_t iClass = classIndicesData[imodel];
        const size_t jClass = classIndicesData[imodel + nModels];

        const bool ijModelNotEmpty(model->getTwoClassClassifierModel(imodel));
        nonEmptyClass[iClass] |= ijModelNotEmpty;
        nonEmptyClass[jClass] |= ijModelNotEmpty;
    }

    size_t nNonEmptyClasses = 0;
    for (size_t i = 0; i < nClasses; i++)
    {
        if (nonEmptyClass[i]) nonEmptyClassMap[nNonEmptyClasses++] = i;
    }
    nClasses = nNonEmptyClasses;
    return services::Status();
}

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif
