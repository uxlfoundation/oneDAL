/* file: elastic_net_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for elastic net classification
//  (defaultDense) method.
//--
*/

#ifndef __ELASTIC_NET_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __ELASTIC_NET_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/elastic_net/elastic_net_train_kernel.h"
#include "src/algorithms/elastic_net/elastic_net_model_impl.h"
#include "src/algorithms/service_error_handling.h"
#include "src/services/service_algo_utils.h"
#include "algorithms/optimization_solver/objective_function/mse_batch.h"
#include "algorithms/optimization_solver/coordinate_descent/coordinate_descent_batch.h"

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "data_management/data/soa_numeric_table.h"
#include "src/externals/service_blas.h"

using namespace daal::algorithms::elastic_net::training::internal;
using namespace daal::algorithms::optimization_solver;
using namespace daal;

namespace daal
{
namespace algorithms
{
namespace elastic_net
{
namespace training
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, elastic_net::training::Method method, CpuType cpu>
services::Status TrainBatchKernel<algorithmFPType, method, cpu>::compute(
    const HostAppIfacePtr & pHost, const NumericTablePtr & x, const NumericTablePtr & y, elastic_net::Model & m, Result & res, const Parameter & par,
    services::SharedPtr<daal::algorithms::optimization_solver::mse::Batch<algorithmFPType> > & objFunc)
{
    services::Status s;
    SafeStatus safeStat;
    const size_t nFeatures           = x->getNumberOfColumns();
    const size_t nRows               = x->getNumberOfRows();
    const size_t p                   = nFeatures + 1;
    const size_t nDependentVariables = m.getBeta()->getNumberOfRows();
    DAAL_ASSERT(p == m.getNumberOfBetas());

    algorithmFPType * xMeansPtr;
    daal::internal::TArray<algorithmFPType, cpu> xMeans;

    algorithmFPType * yMeansPtr;
    daal::internal::TArray<algorithmFPType, cpu> yMeans;

    NumericTablePtr xTrain = x;
    NumericTablePtr yTrain = y;
    if (par.interceptFlag == true)
    {
        if (par.dataUseInComputation == doNotUse)
        {
            int result = 0;
            xTrain     = daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(nFeatures, nRows, &s);
            DAAL_CHECK_STATUS_VAR(s);
            yTrain = daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(nDependentVariables, nRows, &s);
            DAAL_CHECK_STATUS_VAR(s);
            daal::internal::WriteRows<algorithmFPType, cpu> xTrainBD(xTrain.get(), 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(xTrainBD);
            daal::internal::WriteRows<algorithmFPType, cpu> yTrainBD(yTrain.get(), 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(yTrainBD);
            algorithmFPType * xTrainPtr = xTrainBD.get();
            algorithmFPType * yTrainPtr = yTrainBD.get();

            daal::internal::WriteRows<algorithmFPType, cpu> xBD(x.get(), 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(xBD);
            daal::internal::WriteRows<algorithmFPType, cpu> yBD(y.get(), 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(yBD);
            algorithmFPType * const xPtr = xBD.get();
            algorithmFPType * const yPtr = yBD.get();
            result |= daal::services::internal::daal_memcpy_s(xTrainPtr, nFeatures * nRows * sizeof(algorithmFPType), xPtr,
                                                              nFeatures * nRows * sizeof(algorithmFPType));
            result |= daal::services::internal::daal_memcpy_s(yTrainPtr, nDependentVariables * nRows * sizeof(algorithmFPType), yPtr,
                                                              nDependentVariables * nRows * sizeof(algorithmFPType));
            DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
        }

        algorithmFPType inversedNRows = (algorithmFPType)1.0 / (algorithmFPType)nRows;
        yMeans.reset(nDependentVariables);
        yMeansPtr = yMeans.get();
        DAAL_CHECK_MALLOC(yMeansPtr);

        algorithmFPType neg_one = -1.0;
        DAAL_INT one            = 1;
        DAAL_INT zero           = 0;
        DAAL_INT n              = nRows;
        DAAL_INT p              = nFeatures;

        const size_t blockSize = 256;
        size_t nBlocks         = nRows / blockSize;
        nBlocks += (nBlocks * blockSize != nRows);

        StaticTlsSum<algorithmFPType, cpu> yTlsData(nDependentVariables);
        daal::static_threader_for(nBlocks, [&](const size_t iBlock, size_t tid) {
            algorithmFPType * local = yTlsData.local(tid);
            const size_t startRow   = iBlock * blockSize;
            const size_t finishRow  = (iBlock + 1 == nBlocks ? nRows : (iBlock + 1) * blockSize);
            DAAL_INT numRowsInBlock = finishRow - startRow;

            daal::internal::ReadRows<algorithmFPType, cpu> yBD(yTrain.get(), startRow, numRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(yBD);
            const algorithmFPType * yPtr = yBD.get();

            for (size_t i = 0; i < numRowsInBlock; ++i)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t id = 0; id < nDependentVariables; ++id)
                {
                    local[id] += yPtr[i * nDependentVariables + id];
                }
            }
        });
        yTlsData.reduceTo(yMeansPtr, nDependentVariables);

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nDependentVariables; ++i)
        {
            yMeansPtr[i] *= inversedNRows;
        }

        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow   = iBlock * blockSize;
            const size_t finishRow  = (iBlock + 1 == nBlocks ? nRows : (iBlock + 1) * blockSize);
            DAAL_INT numRowsInBlock = finishRow - startRow;

            daal::internal::WriteOnlyRows<algorithmFPType, cpu> yBD(yTrain.get(), startRow, numRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(yBD);
            algorithmFPType * yPtr = yBD.get();

            for (size_t i = 0; i < numRowsInBlock; ++i)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t id = 0; id < nDependentVariables; ++id)
                {
                    yPtr[i * nDependentVariables + id] -= yMeansPtr[id];
                }
            }
        });

        xMeans.reset(nFeatures);
        xMeansPtr = xMeans.get();
        DAAL_CHECK_MALLOC(xMeansPtr);

        for (size_t i = 0; i < nFeatures; ++i) xMeansPtr[i] = 0;

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, sizeof(algorithmFPType));

        StaticTlsSum<algorithmFPType, cpu> tlsData(nFeatures);
        daal::static_threader_for(nBlocks, [&](const size_t iBlock, size_t tid) {
            algorithmFPType * const sum = tlsData.local(tid);
            const size_t startRow       = iBlock * blockSize;
            const size_t finishRow      = (iBlock + 1 == nBlocks ? nRows : (iBlock + 1) * blockSize);
            DAAL_INT numRowsInBlock     = finishRow - startRow;

            daal::internal::ReadRows<algorithmFPType, cpu> xBD(xTrain.get(), startRow, numRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(xBD);
            const algorithmFPType * xPtr = xBD.get();

            for (size_t i = 0; i < numRowsInBlock; ++i)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nFeatures; ++j)
                {
                    sum[j] += xPtr[i * p + j];
                }
            }
        });
        tlsData.reduceTo(xMeansPtr, nFeatures);

        for (size_t i = 0; i < nFeatures; ++i)
        {
            xMeansPtr[i] *= inversedNRows;
        }

        const SOANumericTable * soaPtr = dynamic_cast<SOANumericTable *>(xTrain.get());

        if (soaPtr)
        {
            daal::threader_for(nFeatures, nFeatures, [&](const size_t j) {
                daal::internal::WriteColumns<algorithmFPType, cpu> xBD(xTrain.get(), j, 0, nRows);
                DAAL_CHECK_BLOCK_STATUS_THR(xBD);
                algorithmFPType * xPtr = xBD.get();

                daal::internal::BlasInst<algorithmFPType, cpu>::xxaxpy(&n, &neg_one, xMeansPtr + j, &zero, xPtr, &one);
            });
        }
        else
        {
            daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
                const size_t startRow   = iBlock * blockSize;
                const size_t finishRow  = (iBlock + 1 == nBlocks ? nRows : (iBlock + 1) * blockSize);
                DAAL_INT numRowsInBlock = finishRow - startRow;

                daal::internal::WriteRows<algorithmFPType, cpu> xBD(xTrain.get(), startRow, numRowsInBlock);
                DAAL_CHECK_BLOCK_STATUS_THR(xBD);
                algorithmFPType * xPtr = xBD.get();

                for (size_t j = 0; j < nFeatures; j++)
                {
                    daal::internal::BlasInst<algorithmFPType, cpu>::xxaxpy(&numRowsInBlock, &neg_one, xMeansPtr + j, &zero, xPtr + j, &p);
                }
            });
        }
    }
    services::SharedPtr<optimization_solver::iterative_solver::Batch> pSolver(par.optimizationSolver); //par.optimizationSolver->clone();
    if (!pSolver.get())
    {
        //create cd solver
        services::SharedPtr<optimization_solver::coordinate_descent::Batch<algorithmFPType> > cdAlgorithm =
            optimization_solver::coordinate_descent::Batch<algorithmFPType>::create();
        NumericTablePtr pArg = daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(nDependentVariables, p, &s);
        DAAL_CHECK_STATUS_VAR(s);
        daal::internal::WriteRows<algorithmFPType, cpu> pArgBD(pArg.get(), 0, p);
        DAAL_CHECK_BLOCK_STATUS(pArgBD);
        algorithmFPType * pArgPtr = pArgBD.get();
        daal::services::internal::service_memset<algorithmFPType, cpu>(pArgPtr, 0, nDependentVariables * p);

        cdAlgorithm->input.set(optimization_solver::iterative_solver::inputArgument, pArg);

        cdAlgorithm->parameter().nIterations            = 10000;
        cdAlgorithm->parameter().accuracyThreshold      = 0.00001;
        cdAlgorithm->parameter().selection              = optimization_solver::coordinate_descent::cyclic;
        cdAlgorithm->parameter().positive               = false;
        cdAlgorithm->parameter().skipTheFirstComponents = true;
        pSolver                                         = cdAlgorithm;
    }

    objFunc->input.set(mse::data, xTrain);
    objFunc->input.set(mse::dependentVariables, yTrain);
    objFunc->parameter().interceptFlag = false;

    objFunc->parameter().penaltyL1 = par.penaltyL1;
    objFunc->parameter().penaltyL2 = par.penaltyL2;

    pSolver->getParameter()->function = objFunc;

    if (!(pSolver->getInput()->get(optimization_solver::iterative_solver::inputArgument).get()))
    {
        NumericTablePtr pArg = daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(nDependentVariables, p, &s);
        DAAL_CHECK_STATUS_VAR(s);
        daal::internal::WriteRows<algorithmFPType, cpu> pArgBD(pArg.get(), 0, p);
        DAAL_CHECK_BLOCK_STATUS(pArgBD);
        algorithmFPType * pArgPtr = pArgBD.get();
        daal::services::internal::service_memset<algorithmFPType, cpu>(pArgPtr, 0, nDependentVariables * p);
        DAAL_CHECK_STATUS_VAR(s);
        pSolver->getInput()->set(optimization_solver::iterative_solver::inputArgument, pArg);
    }

    if (!s) return s;
    DAAL_CHECK_STATUS(s, pSolver->compute());

    //write data to model
    daal::internal::ReadRows<algorithmFPType, cpu> ar(*(pSolver->getResult()->get(optimization_solver::iterative_solver::minimum)), 0, p);
    daal::internal::WriteRows<algorithmFPType, cpu> br(*m.getBeta(), 0, nDependentVariables);
    DAAL_CHECK_BLOCK_STATUS(ar);
    DAAL_CHECK_BLOCK_STATUS(br);
    const algorithmFPType * a = ar.get();
    algorithmFPType * pBeta   = br.get();

    for (size_t i = 0; i < nDependentVariables; i++)
    {
        for (size_t j = 1; j < p; j++)
        {
            pBeta[i * p + j] = a[j * nDependentVariables + i];
        }
    }
    if (par.interceptFlag)
    {
        daal::internal::TArray<algorithmFPType, cpu> dotPtr(nDependentVariables);
        algorithmFPType * dot = dotPtr.get();
        for (size_t i = 0; i < nDependentVariables; i++) dot[i] = 0;

        for (size_t i = 0; i < nDependentVariables; i++)
        {
            for (size_t j = 0; j < nFeatures; j++)
            {
                dot[i] += xMeansPtr[j] * pBeta[i * p + j + 1];
            }
        }
        for (size_t j = 0; j < nDependentVariables; ++j) pBeta[p * j + 0] = yMeansPtr[j] - dot[j];
    }
    else
    {
        for (size_t j = 0; j < nDependentVariables; ++j) pBeta[p * j + 0] = 0;
    }

    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace elastic_net */
} /* namespace algorithms */
} /* namespace daal */

#endif
