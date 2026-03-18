/* file: em_gmm_init_dense_default_batch_kernel.h */
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
//  Implementation of em algorithm
//--
*/

#ifndef __EM_GMM_INIT_DENSE_DEFAULT_BATCH_KERNEL_H__
#define __EM_GMM_INIT_DENSE_DEFAULT_BATCH_KERNEL_H__

#include "src/algorithms/kernel.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "src/externals/service_memory.h"
#include "algorithms/em/em_gmm_init_types.h"
#include "algorithms/em/em_gmm_init_batch.h"
#include "algorithms/em/em_gmm.h"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"
#include "src/algorithms/engines/engine.h"
namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace init
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, CpuType cpu>
class GmmSigma
{
    typedef HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef SharedPtr<HomogenNT> HomogenNTPtr;

public:
    GmmSigma(em_gmm::CovarianceStorageId _covType, size_t _nComponents, size_t _nFeatures, Status & st)
        : covType(_covType), nComponents(_nComponents), nFeatures(_nFeatures), sigma(new DataCollection())
    {
        nRows = nFeatures;
        if (covType == em_gmm::diagonal)
        {
            nRows = 1;
        }
        for (size_t i = 0; i < nComponents; i++)
        {
            sigma->push_back(HomogenNT::create(nFeatures, nRows, &st));
            if (!st) return;
        }
    }
    void setVariance(algorithmFPType * varianceArray)
    {
        for (size_t k = 0; k < nComponents; k++)
        {
            auto workSigma               = static_cast<HomogenNT *>((*sigma)[k].get());
            algorithmFPType * sigmaArray = workSigma->getArray();
            if (covType == em_gmm::diagonal)
            {
                for (size_t i = 0; i < nFeatures; i++)
                {
                    sigmaArray[i] = varianceArray[i];
                }
            }
            else
            {
                for (size_t i = 0; i < nFeatures * nFeatures; i++)
                {
                    sigmaArray[i] = 0.0;
                }
                for (size_t i = 0; i < nFeatures; i++)
                {
                    sigmaArray[i * nFeatures + i] = varianceArray[i];
                }
            }
        }
    }
    DataCollectionPtr & getSigma() { return sigma; }
    Status writeToTables(DataCollectionPtr covariancesToInit)
    {
        for (size_t k = 0; k < nComponents; k++)
        {
            NumericTablePtr covariance = staticPointerCast<NumericTable, SerializationIface>((*covariancesToInit)[k]);
            WriteOnlyRows<algorithmFPType, cpu, NumericTable> covarianceBlock(covariance.get(), 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(covarianceBlock)
            algorithmFPType * covarianceArray = covarianceBlock.get();

            auto workSigma = static_cast<HomogenNT *>((*sigma)[k].get());
            for (size_t i = 0; i < nRows * nFeatures; i++)
            {
                covarianceArray[i] = (workSigma->getArray())[i];
            }
        }
        return Status();
    }

private:
    DataCollectionPtr sigma;
    em_gmm::CovarianceStorageId covType;
    size_t nFeatures;
    size_t nComponents;
    size_t nRows;
};

template <typename algorithmFPType, Method method, CpuType cpu>
class EMInitKernel : public Kernel
{
public:
    services::Status compute(NumericTable & data, NumericTable & weightsToInit, NumericTable & meansToInit, DataCollectionPtr & covariancesToInit,
                             const Parameter & par, engines::BatchBase & engine);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class EMInitKernelTask
{
    typedef HomogenNumericTableCPU<algorithmFPType, cpu> HomogenNT;
    typedef SharedPtr<HomogenNT> HomogenNTPtr;

public:
    EMInitKernelTask(NumericTable & data, NumericTable & weightsToInit, NumericTable & meansToInit, DataCollectionPtr & covariancesToInit,
                     const Parameter & parameter, engines::BatchBase & engine, Status & status);
    Status compute();

private:
    Status writeValuesToTables();
    Status setSelectedSetAsInitialValues();
    ErrorID runEM();
    Status generateSelectedSet();
    Status initialize();
    Status computeVariance();

    NumericTable & data;
    NumericTable & weightsToInit;
    NumericTable & meansToInit;
    DataCollectionPtr & covariancesToInit;
    const Parameter & parameter;
    const size_t nComponents;
    const size_t nFeatures;
    const size_t nVectors;
    const size_t nTrials;
    const size_t nIterations;
    double accuracyThreshold;
    HomogenNTPtr alpha;
    HomogenNTPtr means;
    algorithmFPType loglikelyhood;
    algorithmFPType maxLoglikelyhood;
    algorithmFPType * varianceArray;
    TArray<algorithmFPType, cpu> varianceArrayPtr;
    TArray<int, cpu> selectedSetPtr;
    int * selectedSet;
    GmmSigma<algorithmFPType, cpu> covs;
    engines::BatchBase & engine;
};

template <typename algorithmFPType>
class EMforKernel : public daal::algorithms::em_gmm::Batch<algorithmFPType, em_gmm::defaultDense>
{
public:
    EMforKernel(const size_t nComponents) : daal::algorithms::em_gmm::Batch<algorithmFPType, em_gmm::defaultDense>(nComponents) {}
    virtual ~EMforKernel() {}

    ErrorID run(data_management::NumericTable & inputData, data_management::NumericTable & inputWeights, data_management::NumericTable & inputMeans,
                data_management::DataCollectionPtr & inputCov, const em_gmm::CovarianceStorageId covType, algorithmFPType & loglikelyhood);
};

} // namespace internal

} // namespace init

} // namespace em_gmm

} // namespace algorithms

} // namespace daal

#endif
