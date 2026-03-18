/* file: multiclassclassifier_train_batch_container.h */
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
//  Implementation of Multi-class classifier algorithm container -- a class that contains
//  Multi-class classifier kernels for supported architectures.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_train.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_train_kernel.h"
#include "src/algorithms/multiclassclassifier/multiclassclassifier_train_oneagainstone_kernel.h"
#include "src/algorithms/kernel.h"

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
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__TRAINING__BATCHCONTAINER"></a>
 * \brief Class containing methods to compute the results of multi-class classifier model-based training
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the multi-class classifier, double or float
 * \tparam method           Computation method of the algprithm, \ref daal::algorithms::multi_class_classifier::training::Method
 *
 */
template <typename algorithmFPType, Method method, CpuType cpu>
class BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for multi-class classifier model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of multi-class classifier model-based training in the batch processing mode
     *
     * \return Status of computation
     */
    services::Status compute() override;
};

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::MultiClassClassifierTrainKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    const classifier::training::Input * input = static_cast<const classifier::training::Input *>(_in);
    Result * result                           = static_cast<Result *>(_res);

    const NumericTable * a[3];
    a[0] = static_cast<NumericTable *>(input->get(classifier::training::data).get());
    a[1] = static_cast<NumericTable *>(input->get(classifier::training::labels).get());
    a[2] = static_cast<NumericTable *>(input->get(classifier::training::weights).get());

    multi_class_classifier::Model * r = static_cast<multi_class_classifier::Model *>(result->get(classifier::training::model).get());

    const multi_class_classifier::Parameter * par = static_cast<const multi_class_classifier::Parameter *>(_par);

    internal::KernelParameter kernelPar;
    kernelPar.nClasses          = par->nClasses;
    kernelPar.training          = par->training;
    kernelPar.prediction        = par->prediction;
    kernelPar.maxIterations     = par->maxIterations;
    kernelPar.accuracyThreshold = par->accuracyThreshold;

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::MultiClassClassifierTrainKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, a[0], a[1], a[2], r,
                       nullptr, kernelPar);
}

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
