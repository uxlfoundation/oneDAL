/* file: svm_train_internal.h */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef __SVM_TRAIN_INTERNAL_H__
#define __SVM_TRAIN_INTERNAL_H__

#include "algorithms/algorithm.h"

#include "algorithms/svm/svm_train_types.h"
#include "src/algorithms/svm/svm_train_kernel.h"
#include "algorithms/classifier/classifier_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class NuBatchContainer : public TrainingContainerIface<batch>
{
public:
    NuBatchContainer(daal::services::Environment::env * daalEnv);

    ~NuBatchContainer();

    services::Status compute() override;
};

/**
*  \brief Initialize list of SVM kernels with implementations for supported
* architectures
*/
template <typename algorithmFPType, Method method, CpuType cpu>
NuBatchContainer<algorithmFPType, method, cpu>::NuBatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::SVMTrainImpl, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
NuBatchContainer<algorithmFPType, method, cpu>::~NuBatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status NuBatchContainer<algorithmFPType, method, cpu>::compute()
{
    classifier::training::Input * input = static_cast<classifier::training::Input *>(_in);
    svm::training::Result * result      = static_cast<svm::training::Result *>(_res);

    const NumericTablePtr x       = input->get(classifier::training::data);
    const NumericTablePtr y       = input->get(classifier::training::labels);
    const NumericTablePtr weights = input->get(classifier::training::weights);

    daal::algorithms::Model * r = static_cast<daal::algorithms::Model *>(result->get(classifier::training::model).get());

    internal::KernelParameter kernelPar = *static_cast<internal::KernelParameter *>(_par);

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::SVMTrainImpl, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, x, weights, *y, r, kernelPar);
}

template <typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = boser>
class DAAL_EXPORT Batch : public classifier::training::Batch
{
public:
    typedef classifier::training::Batch super;

    typedef typename super::InputType InputType;
    typedef KernelParameter ParameterType;
    typedef algorithms::svm::training::Result ResultType;

    ParameterType parameter;
    InputType input;

    Batch();

    Batch(size_t nClasses);

    Batch(const Batch<algorithmFPType, method> & other);

    virtual ~Batch() {}

    InputType * getInput() override { return &input; }

    virtual int getMethod() const override { return (int)method; }

    ResultPtr getResult() { return ResultType::cast(_result); }

    services::Status resetResult() override
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult);
        _res = NULL;
        return services::Status();
    }

    services::SharedPtr<Batch<algorithmFPType, method> > clone() const { return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl()); }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const override { return new Batch<algorithmFPType, method>(*this); }

    services::Status allocateResult() override
    {
        ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, _par, (int)method);
        _res               = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, NuBatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    Batch & operator=(const Batch &);
};

} // namespace internal

} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
