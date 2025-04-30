/* file: adagrad_dense_default_impl.i */
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
//  Implementation of adagrad algorithm
//--
*/

#ifndef __ADAGRAD_DENSE_DEFAULT_IMPL_I__
#define __ADAGRAD_DENSE_DEFAULT_IMPL_I__

#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "algorithms/optimization_solver/adagrad/adagrad_types.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace adagrad
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::optimization_solver::iterative_solver::internal;

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status AdagradKernel<algorithmFPType, method, cpu>::initAccumulatedGrad(algorithmFPType * accumulatedG, size_t nRows,
                                                                                  NumericTable * pOptInput)
{
    int result = 0;
    if (pOptInput)
    {
        /* Initialize accumulatedG and invAccumGrad from optional input data */
        ReadRows<algorithmFPType, cpu> optInputBD(*pOptInput, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(optInputBD);
        const algorithmFPType * optInputArray = optInputBD.get();
        processByBlocks<cpu>(
            nRows,
            [=, &result](size_t startOffset, size_t nRowsInBlock) {
                result |= daal::services::internal::daal_memcpy_s(&accumulatedG[startOffset], nRowsInBlock * sizeof(algorithmFPType),
                                                                  &optInputArray[startOffset], nRowsInBlock * sizeof(algorithmFPType));
            },
            _blockSize, _threadStart);
    }
    else
    {
        processByBlocks<cpu>(
            nRows,
            [=](size_t startOffset, size_t nRowsInBlock) {
                for (size_t i = startOffset; i < startOffset + nRowsInBlock; i++)
                {
                    accumulatedG[i] = algorithmFPType(0.0);
                }
            },
            _blockSize, _threadStart);
    }
    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}

/**
 *  \brief Kernel for Adagrad calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status AdagradKernel<algorithmFPType, method, cpu>::compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum,
                                                                      NumericTable * nIterations, NumericTable * gradientSquareSumResult,
                                                                      NumericTable * gradientSquareSumInput, OptionalArgument * optionalArgument,
                                                                      OptionalArgument * optionalResult, Parameter * parameter,
                                                                      engines::BatchBase & engine)
{
    const size_t nRows = inputArgument->getNumberOfRows();
    int result         = 0;

    WriteRows<algorithmFPType, cpu> workValueBD(*minimum, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(workValueBD);
    algorithmFPType * workValue = workValueBD.get();

    //init workValue
    {
        ReadRows<algorithmFPType, cpu> startValueBD(*inputArgument, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(startValueBD);
        const algorithmFPType * startValueArray = startValueBD.get();
        processByBlocks<cpu>(
            nRows,
            [=, &result](size_t startOffset, size_t nRowsInBlock) {
                result |= daal::services::internal::daal_memcpy_s(&workValue[startOffset], nRowsInBlock * sizeof(algorithmFPType),
                                                                  &startValueArray[startOffset], nRowsInBlock * sizeof(algorithmFPType));
            },
            _blockSize, _threadStart);
    }
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

    const size_t nIter = parameter->nIterations;
    WriteRows<int, cpu> nIterationsBD(*nIterations, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(nIterationsBD);
    int * nProceededIterations = nIterationsBD.get();
    /* if nIter == 0, set result as start point, the number of executed iters to 0 */
    if (nIter == 0)
    {
        nProceededIterations[0] = 0;
        return Status();
    }

    NumericTable * lastIterationInput =
        (optionalArgument) ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;
    NumericTable * lastIterationResult = (optionalResult) ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr;

    sum_of_functions::BatchPtr function = parameter->function;

    /*Get random indices for SGD from parameter or from rng generator*/
    const size_t batchSize                         = parameter->batchSize;
    const algorithmFPType degenerateCasesThreshold = (algorithmFPType)parameter->degenerateCasesThreshold;
    ReadRows<int, cpu> predefinedBatchIndicesBD(parameter->batchIndices.get(), 0, nIter);
    DAAL_CHECK_BLOCK_STATUS(predefinedBatchIndicesBD);
    RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    if (!parameter->batchIndices)
    {
        DAAL_CHECK_MALLOC(rngTask.init(parameter->function->sumOfFunctionsParameter->numberOfTerms, engine));
    }

    services::Status s;
    SharedPtr<HomogenNumericTableCPU<int, cpu> > ntBatchIndices = HomogenNumericTableCPU<int, cpu>::create(NULL, batchSize, 1, &s);
    DAAL_CHECK_STATUS_VAR(s);
    function->sumOfFunctionsParameter->batchIndices = ntBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, HomogenNumericTableCPU<algorithmFPType, cpu>::create(workValue, 1, nRows, &s));
    DAAL_CHECK_STATUS_VAR(s);

    NumericTablePtr ntlearningRate = parameter->learningRate;
    ReadRows<algorithmFPType, cpu> learningRateBD(*ntlearningRate, 0, 1);
    const algorithmFPType learningRate = *learningRateBD.get();

    DAAL_CHECK(nIter <= services::internal::MaxVal<int>::get(), ErrorIterativeSolverIncorrectMaxNumberOfIterations)
    *nProceededIterations = (int)nIter;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> smAccumulatedG(nRows);
    DAAL_CHECK_MALLOC(smAccumulatedG.get());
    algorithmFPType * accumulatedG = (algorithmFPType *)smAccumulatedG.get();

    s = initAccumulatedGrad(accumulatedG, nRows, gradientSquareSumInput);
    DAAL_CHECK_STATUS_VAR(s);

    size_t startIteration = 0, nProceededIters = 0;
    if (lastIterationInput != nullptr)
    {
        ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
        const int * lastIterationInputArray = lastIterationInputBD.get();
        startIteration                      = lastIterationInputArray[0];
    }
    services::internal::HostAppHelper host(pHost, 10);
    for (size_t epoch = startIteration; epoch < (startIteration + nIter); epoch++)
    {
        const int * pValues = nullptr;
        s                   = rngTask.get(pValues);
        if (s)
        {
            ntBatchIndices->setArray(const_cast<int *>(pValues), ntBatchIndices->getNumberOfRows());
            s = function->computeNoThrow();
        }
        if (!s || host.isCancelled(s, 1))
        {
            DAAL_ASSERT((epoch - startIteration) <= services::internal::MaxVal<int>::get())
            *nProceededIterations = (int)(epoch - startIteration);
            break;
        }

        auto ntGradient = function->getResult()->get(objective_function::gradientIdx);
        ReadRows<algorithmFPType, cpu> ntGradientBD(*ntGradient, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(ntGradientBD);
        const algorithmFPType * gradient = ntGradientBD.get();
        if (nIter > 1)
        {
            algorithmFPType pointNorm, gradientNorm;
            DAAL_CHECK_STATUS(s, (IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm(workValue, nRows, pointNorm, _blockSize, _threadStart)));
            DAAL_CHECK_STATUS(s, (IterativeSolverKernel<algorithmFPType, cpu>::vectorNorm(gradient, nRows, gradientNorm, _blockSize, _threadStart)));
            const algorithmFPType gradientThreshold =
                parameter->accuracyThreshold * daal::internal::MathInst<algorithmFPType, cpu>::sMax(algorithmFPType(1.0), pointNorm);
            if (gradientNorm <= gradientThreshold)
            {
                DAAL_ASSERT((epoch - startIteration) <= services::internal::MaxVal<int>::get())
                *nProceededIterations = (int)(epoch - startIteration);
                if (parameter->optionalResultRequired)
                {
                    processByBlocks<cpu>(
                        nRows,
                        [=](size_t startOffset, size_t nRowsInBlock) {
                            PRAGMA_FORCE_SIMD
                            PRAGMA_VECTOR_ALWAYS
                            for (size_t j = startOffset; j < startOffset + nRowsInBlock; j++)
                            {
                                accumulatedG[j] += gradient[j] * gradient[j];
                            }
                        },
                        _blockSize, _threadStart);
                }
                break;
            }
        }

        processByBlocks<cpu>(
            nRows,
            [=](size_t startOffset, size_t nRowsInBlock) {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = startOffset; j < startOffset + nRowsInBlock; j++)
                {
                    accumulatedG[j] += gradient[j] * gradient[j];
                    const algorithmFPType invAccumGrad =
                        algorithmFPType(1.0) / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(accumulatedG[j] + degenerateCasesThreshold);
                    workValue[j] = workValue[j] - learningRate * gradient[j] * invAccumGrad;
                }
            },
            _blockSize, _threadStart);
        nProceededIters++;
    }
    if (!s || !parameter->optionalResultRequired) return s;
    NumericTable * pOptResult = gradientSquareSumResult;
    /* Copy accumulatedG to output */
    WriteRows<algorithmFPType, cpu> optResultBD(*pOptResult, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(optResultBD);
    result |=
        daal::services::internal::daal_memcpy_s(optResultBD.get(), nRows * sizeof(algorithmFPType), accumulatedG, nRows * sizeof(algorithmFPType));
    if (lastIterationResult)
    {
        WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult, 0, 1);
        int * lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0]    = startIteration + nProceededIters;
    }
    return (!result) ? s : Status(ErrorMemoryCopyFailedInternal);
}

} // namespace internal

} // namespace adagrad

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
