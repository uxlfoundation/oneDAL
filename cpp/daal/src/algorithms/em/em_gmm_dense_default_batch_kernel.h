/* file: em_gmm_dense_default_batch_kernel.h */
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
//  Declaration of template function that calculate ems.
//--
*/

#ifndef __EM_GMM_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __EM_GMM_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "algorithms/em/em_gmm.h"
#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "src/externals/service_blas.h"
#include "src/algorithms/em/em_gmm_dense_default_batch_task.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class EMKernel : public Kernel
{
public:
    typedef SharedPtr<GmmModel<algorithmFPType, cpu> > GmmModelPtr;
    EMKernel() {};

    services::Status compute(NumericTable & dataTable, NumericTable & initialWeights, NumericTable & initialMeans, NumericTable ** initialCovariances,
                             NumericTable & resultWeights, NumericTable & resultMeans, NumericTable ** resultCovariances,
                             NumericTable & resultNIterations, NumericTable & resultGoalFunction, const Parameter & par);
};

template <typename algorithmFPType, CpuType cpu>
void stepM_mergePartialSums(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n, algorithmFPType * mean_m, algorithmFPType & w_n,
                            algorithmFPType & w_m, size_t nFeatures, GmmModel<algorithmFPType, cpu> * covs);

template <typename algorithmFPType, Method method, CpuType cpu>
class EMKernelTask
{
    typedef SharedPtr<GmmModel<algorithmFPType, cpu> > GmmModelPtr;
    typedef GmmModelDiag<algorithmFPType, cpu> GmmModelDiagType;
    typedef GmmModelFull<algorithmFPType, cpu> GmmModelFullType;

    SharedPtr<GmmModel<algorithmFPType, cpu> > initializeCovariances();

public:
    EMKernelTask(NumericTable & dataTable, NumericTable & initialWeights, NumericTable & initialMeans, NumericTable ** initialCovariances,
                 NumericTable & resultWeights, NumericTable & resultMeans, NumericTable ** resultCovariances, NumericTable & resultNIterations,
                 NumericTable & resultGoalFunction, const Parameter & par);

    services::Status compute();

    Status initialize();
    services::Status setStartValues();
    void setResultToZero();
    Status stepM_merge(size_t iteration);

    static void stepE(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> & t, em_gmm::CovarianceStorageId covType);
    static algorithmFPType computePartialLogLikelihood(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> & t);
    static Status stepM_partial(const size_t nVectorsInCurrentBlock, Task<algorithmFPType, cpu> & t, em_gmm::CovarianceStorageId covType);
    static void stepM_mergePartialSums(algorithmFPType * cp_n, algorithmFPType * cp_m, algorithmFPType * mean_n, algorithmFPType * mean_m,
                                       algorithmFPType & w_n, algorithmFPType & w_m, size_t nFeatures, GmmModel<algorithmFPType, cpu> * covs);

    algorithmFPType * alpha;
    algorithmFPType * means;
    algorithmFPType * logAlpha;
    int * iterCounterArray;
    algorithmFPType * logLikelihoodArray;

    size_t blockSizeDefault;
    size_t nBlocks;

    const DAAL_INT nFeatures;
    const DAAL_INT nVectors;
    const DAAL_INT nComponents;
    algorithmFPType logLikelihoodCorrection;
    const DAAL_INT maxIterations;
    const algorithmFPType threshold;
    TArray<WriteRows<algorithmFPType, cpu, NumericTable>, cpu> covsPtr;
    GmmModelPtr covs;

    WriteRows<algorithmFPType, cpu, NumericTable> weightsBD;
    WriteRows<algorithmFPType, cpu, NumericTable> meansBD;
    WriteRows<int, cpu, NumericTable> nIterationsBD;
    WriteRows<algorithmFPType, cpu, NumericTable> goalFunctionBD;

    NumericTable & dataTable;
    NumericTable & initialWeights;
    NumericTable & initialMeans;
    NumericTable ** initialCovariances;
    NumericTable & resultWeights;
    NumericTable & resultMeans;
    NumericTable ** resultCovariances;
    NumericTable & resultNIterations;
    NumericTable & resultGoalFunction;
    const Parameter & par;
};

} // namespace internal

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
