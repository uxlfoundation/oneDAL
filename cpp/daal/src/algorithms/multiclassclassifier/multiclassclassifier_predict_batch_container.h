/* file: multiclassclassifier_predict_batch_container.h */
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
//  Implementation of Multi-class classifier prediction algorithm container --
//  a class that contains Multi-class classifier kernels for supported
//  architectures.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_predict.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_predict_kernel.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_predict_mccwu_kernel.h"
#include "src/algorithms/kernel.h"

using namespace daal::data_management;

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
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the  multi-class classifier prediction algorithm
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the multi-class classifier algorithm, double or float
 * \tparam pmethod          Computation method for the algorithm, \ref prediction::Method
 * \tparam tmethod          Computation method that was used to train the multi-class classifier model, \ref training::Method
 *
 */
template <typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
class BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for multi-class classifier model-based prediction with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of multi-class classifier model-based prediction in the batch processing mode
     *
     * \return Status of computation
     */
    services::Status compute() override;
};

template <typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MultiClassClassifierPredictKernel, pmethod, tmethod, algorithmFPType);
}

template <typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, prediction::Method pmethod, training::Method tmethod, CpuType cpu>
services::Status BatchContainer<algorithmFPType, pmethod, tmethod, cpu>::compute()
{
    const classifier::prediction::Input * input = static_cast<const classifier::prediction::Input *>(_in);

    NumericTable * r[2];
    Result * result = dynamic_cast<Result *>(_res);

    if (result)
    {
        r[0] = static_cast<NumericTable *>(result->get(ResultId::prediction).get());
        r[1] = static_cast<NumericTable *>(result->get(ResultId::decisionFunction).get());
    }
    else
    {
        // for static BC
        classifier::prediction::Result * resultCls = static_cast<classifier::prediction::Result *>(_res);
        r[0]                                       = static_cast<NumericTable *>(resultCls->get(classifier::prediction::ResultId::prediction).get());
        r[1]                                       = nullptr;
    }

    const NumericTable * a                  = static_cast<NumericTable *>(input->get(classifier::prediction::data).get());
    const multi_class_classifier::Model * m = static_cast<const multi_class_classifier::Model *>(input->get(classifier::prediction::model).get());

    const daal::algorithms::Parameter * par = _par;
    daal::services::Environment::env & env  = *_env;
    __DAAL_CALL_KERNEL(env, internal::MultiClassClassifierPredictKernel, __DAAL_KERNEL_ARGUMENTS(pmethod, tmethod, algorithmFPType), compute, a, m,
                       nullptr, r[0], r[1], par);
}

} // namespace internal
} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
