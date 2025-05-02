/* file: sgd_dense_minibatch_impl.i */
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
//  Implementation of sgd miniBatch algorithm
//
// Mu Li, Tong Zhang, Yuqiang Chen, Alexander J. Smola Efficient Mini-batch Training for Stochastic Optimization
//--
*/

#ifndef __SGD_DENSE_MINIBATCH_IMPL_I__
#define __SGD_DENSE_MINIBATCH_IMPL_I__

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_profiler.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sgd
{
namespace internal
{
/**
 *  \brief Kernel for SGD miniBatch calculation
 */
template <typename algorithmFPType, CpuType cpu>
services::Status SGDKernel<algorithmFPType, miniBatch, cpu>::compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum,
                                                                     NumericTable * nIterations, Parameter<miniBatch> * parameter,
                                                                     NumericTable * learningRateSequence, NumericTable * batchIndices,
                                                                     OptionalArgument * optionalArgument, OptionalArgument * optionalResult,
                                                                     engines::BatchBase & engine)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(SGDKernel(miniBatch).compute);

    services::Status s;
    int result                = 0;
    const size_t argumentSize = inputArgument->getNumberOfRows();
    const size_t nIter        = parameter->nIterations;
    const size_t L            = parameter->innerNIterations;
    const size_t batchSize    = parameter->batchSize;

    /* if nIter == 0, set result as start point, the number of executed iters to 0 */
    if (nIter == 0 || L == 0)
    {
        SGDMiniBatchTask<algorithmFPType, cpu> task(argumentSize, minimum, inputArgument, nIterations);
        DAAL_CHECK_STATUS(s, task.init(inputArgument));
        task.nProceededIterations[0] = 0;
        return Status();
    }

    SharedPtr<sum_of_functions::Batch> function = parameter->function;
    const size_t nTerms                         = function->sumOfFunctionsParameter->numberOfTerms;
    NumericTable * conservativeSequence         = parameter->conservativeSequence.get();

    SGDMiniBatchTask<algorithmFPType, cpu> task(
        batchSize, argumentSize, nIter, nTerms, minimum, inputArgument, learningRateSequence, conservativeSequence, nIterations, batchIndices,
        optionalResult ? NumericTable::cast(optionalResult->get(sgd::pastWorkValue)).get() : nullptr,
        optionalResult ? NumericTable::cast(optionalResult->get(iterative_solver::lastIteration)).get() : nullptr);

    DAAL_CHECK_STATUS(s, task.init(inputArgument, learningRateSequence, conservativeSequence, nIterations, batchIndices, optionalArgument));
    if (!task.ntBatchIndices)
    {
        return Status(ErrorIncorrectParameter);
    }
    algorithmFPType * workValue = task.mtWorkValue.get();

    NumericTablePtr previousArgument = function->sumOfFunctionsInput->get(sum_of_functions::argument);
    function->sumOfFunctionsInput->set(sum_of_functions::argument, task.ntWorkValue);

    double accuracyThreshold = parameter->accuracyThreshold;

    ReadRows<algorithmFPType, cpu> gradientBlock;
    NumericTablePtr ntGradient;
    algorithmFPType learningRate, consCoeff;

    NumericTablePtr previousBatchIndices            = function->sumOfFunctionsParameter->batchIndices;
    function->sumOfFunctionsParameter->batchIndices = task.ntBatchIndices;

    ReadRows<int, cpu> predefinedBatchIndicesBD(batchIndices, 0, nIter);
    using namespace iterative_solver::internal;
    RngTask<int, cpu> rngTask(predefinedBatchIndicesBD.get(), batchSize);
    DAAL_CHECK_MALLOC(batchIndices || rngTask.init(nTerms, engine));

    services::internal::HostAppHelper host(pHost, 10);
    for (size_t epoch = task.startIteration; s.ok() && (epoch < (task.startIteration + nIter)); epoch++)
    {
        if (epoch % L == 0 || epoch == task.startIteration)
        {
            learningRate = task.learningRateArray[(epoch / L) % task.learningRateLength];
            consCoeff    = task.consCoeffsArray[(epoch / L) % task.consCoeffsLength];
            if (task.indicesStatus == user || task.indicesStatus == random)
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(generateUniform);
                const int * pValues = nullptr;
                s                   = rngTask.get(pValues);
                DAAL_CHECK_BREAK(!s);
                task.ntBatchIndices->setArray(const_cast<int *>(pValues), task.ntBatchIndices->getNumberOfRows());
            }
        }
        s = function->computeNoThrow();
        if (!s || host.isCancelled(s, 1))
        {
            DAAL_ASSERT((epoch - task.startIteration) <= services::internal::MaxVal<int>::get())
            task.nProceededIterations[0] = (int)(epoch - task.startIteration);
            break;
        }

        ntGradient = function->getResult()->get(objective_function::gradientIdx);
        gradientBlock.set(*ntGradient, 0, argumentSize);
        if (!gradientBlock.status())
        {
            s = gradientBlock.status();
            break;
        }
        const algorithmFPType * gradient = gradientBlock.get();

        if (epoch % L == 0)
        {
            if (nIter > 1)
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(convergence_check);

                algorithmFPType pointNorm, gradientNorm;
                s = vectorNorm(workValue, argumentSize, pointNorm);
                s |= vectorNorm(gradient, argumentSize, gradientNorm);
                DAAL_CHECK_BREAK(!s);
                double gradientThreshold = accuracyThreshold * daal::internal::MathInst<algorithmFPType, cpu>::sMax(1.0, pointNorm);
                DAAL_CHECK_BREAK(gradientNorm < gradientThreshold);
            }
            result |= daal::services::internal::daal_memcpy_s(task.prevWorkValue.get(), argumentSize * sizeof(algorithmFPType), workValue,
                                                              argumentSize * sizeof(algorithmFPType));
        }
        task.makeStep(gradient, learningRate, consCoeff, argumentSize);
    }
    DAAL_CHECK(task.nProceededIters <= services::internal::MaxVal<int>::get(), ErrorIterativeSolverIncorrectMaxNumberOfIterations)
    task.nProceededIterations[0]                    = (int)task.nProceededIters;
    function->sumOfFunctionsParameter->batchIndices = previousBatchIndices;
    function->sumOfFunctionsInput->set(sum_of_functions::argument, previousArgument);

    return (!result) ? s : Status(ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
void SGDMiniBatchTask<algorithmFPType, cpu>::makeStep(const algorithmFPType * gradient, algorithmFPType learningRate, algorithmFPType consCoeff,
                                                      size_t argumentSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(makeStep);

    algorithmFPType * workValue = mtWorkValue.get();
    for (size_t j = 0; j < argumentSize; j++)
    {
        workValue[j] = workValue[j] - learningRate * (gradient[j] + consCoeff * (workValue[j] - prevWorkValue[j]));
    }
    nProceededIters++;
}

template <typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(size_t argumentSize_, NumericTable * resultTable, NumericTable * startValueTable,
                                                         NumericTable * nIterationsTable)
    : argumentSize(argumentSize_),
      mtWorkValue(resultTable, 0, argumentSize),
      mtNIterations(nIterationsTable, 0, 1),
      nProceededIterations(nullptr),
      learningRateArray(nullptr),
      consCoeffsArray(nullptr)
{}

template <typename algorithmFPType, CpuType cpu>
services::Status SGDMiniBatchTask<algorithmFPType, cpu>::init(NumericTable * startValueTable)
{
    DAAL_CHECK_BLOCK_STATUS(mtNIterations);
    nProceededIterations = mtNIterations.get();
    return setStartValue(startValueTable);
}

template <typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::~SGDMiniBatchTask()
{
    if (lastIterationResult)
    {
        WriteRows<int, cpu, NumericTable> lastIterationResultBD(lastIterationResult.get(), 0, 1);
        int * lastIterationResultArray = lastIterationResultBD.get();
        lastIterationResultArray[0]    = startIteration + nProceededIters;
    }

    if (pastWorkValueResult)
    {
        WriteRows<algorithmFPType, cpu, NumericTable> pastWorkValueResultBD(pastWorkValueResult.get(), 0, pastWorkValueResult->getNumberOfRows());
        algorithmFPType * pastWorkValueResultArray = pastWorkValueResultBD.get();
        int result = daal::services::internal::daal_memcpy_s(pastWorkValueResultArray, argumentSize * sizeof(algorithmFPType), prevWorkValue.get(),
                                                             argumentSize * sizeof(algorithmFPType));
        _status |= (result) ? services::Status(services::ErrorMemoryCopyFailedInternal) : services::Status();
    }
}

template <typename algorithmFPType, CpuType cpu>
Status SGDMiniBatchTask<algorithmFPType, cpu>::setStartValue(NumericTable * startValueTable)
{
    DAAL_CHECK_BLOCK_STATUS(mtWorkValue);
    ReadRows<algorithmFPType, cpu> mtStartValue(startValueTable, 0, argumentSize);
    DAAL_CHECK_BLOCK_STATUS(mtStartValue);
    int result = daal::services::internal::daal_memcpy_s(mtWorkValue.get(), argumentSize * sizeof(algorithmFPType), mtStartValue.get(),
                                                         argumentSize * sizeof(algorithmFPType));
    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
SGDMiniBatchTask<algorithmFPType, cpu>::SGDMiniBatchTask(size_t batchSize_, size_t argumentSize_, size_t nIter_, size_t nTerms_,
                                                         NumericTable * resultTable, NumericTable * startValueTable,
                                                         NumericTable * learningRateSequenceTable, NumericTable * conservativeSequenceTable,
                                                         NumericTable * nIterationsTable, NumericTable * batchIndicesTable,
                                                         NumericTable * pastWorkValueResultNT, NumericTable * lastIterationResultNT)
    : batchSize(batchSize_),
      argumentSize(argumentSize_),
      nIter(nIter_),
      nTerms(nTerms_),
      mtWorkValue(resultTable, 0, argumentSize),
      mtLearningRate(learningRateSequenceTable, 0, learningRateSequenceTable->getNumberOfRows()),
      mtConsCoeffs(conservativeSequenceTable, 0, conservativeSequenceTable->getNumberOfRows()),
      mtNIterations(nIterationsTable, 0, 1),
      mtPredefinedBatchIndices(batchIndicesTable),
      nProceededIterations(nullptr),
      learningRateArray(nullptr),
      consCoeffsArray(nullptr),
      prevWorkValue(argumentSize),
      lastIterationResult(lastIterationResultNT, EmptyDeleter()),
      pastWorkValueResult(pastWorkValueResultNT, EmptyDeleter()),
      startIteration(0),
      nProceededIters(0)
{}

template <typename algorithmFPType, CpuType cpu>
Status SGDMiniBatchTask<algorithmFPType, cpu>::init(NumericTable * startValueTable, NumericTable * learningRateSequenceTable,
                                                    NumericTable * conservativeSequenceTable, NumericTable * nIterationsTable,
                                                    NumericTable * batchIndicesTable, OptionalArgument * optionalArgument)
{
    Status s   = setStartValue(startValueTable);
    int result = 0;
    DAAL_CHECK_STATUS_VAR(s);
    algorithmFPType * workValue = mtWorkValue.get();
    ntWorkValue.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, 1, argumentSize, s));
    DAAL_CHECK_MALLOC(ntWorkValue.get());

    DAAL_CHECK_BLOCK_STATUS(mtLearningRate);
    learningRateArray  = mtLearningRate.get();
    learningRateLength = learningRateSequenceTable->getNumberOfColumns();

    DAAL_CHECK_BLOCK_STATUS(mtConsCoeffs);
    consCoeffsArray  = mtConsCoeffs.get();
    consCoeffsLength = conservativeSequenceTable->getNumberOfColumns();

    DAAL_CHECK_BLOCK_STATUS(mtNIterations);
    nProceededIterations    = mtNIterations.get();
    nProceededIterations[0] = 0;

    DAAL_CHECK_MALLOC(prevWorkValue.get());

    indicesStatus = (batchIndicesTable ? user : (batchSize < nTerms ? random : all));
    if (indicesStatus == user || indicesStatus == random)
    {
        ntBatchIndices.reset(new HomogenNumericTableCPU<int, cpu>(NULL, batchSize, 1, s));
        DAAL_CHECK_MALLOC(ntBatchIndices.get());
    }

    NumericTable * pastWorkValueInput = optionalArgument ? NumericTable::cast(optionalArgument->get(sgd::pastWorkValue)).get() : nullptr;
    NumericTable * lastIterationInput = optionalArgument ? NumericTable::cast(optionalArgument->get(iterative_solver::lastIteration)).get() : nullptr;

    if (lastIterationInput)
    {
        ReadRows<int, cpu, NumericTable> lastIterationInputBD(lastIterationInput, 0, 1);
        const int * lastIterationInputArray = lastIterationInputBD.get();
        startIteration                      = lastIterationInputArray[0];
    }

    if (pastWorkValueInput)
    {
        ReadRows<algorithmFPType, cpu, NumericTable> pastWorkValueInputBD(pastWorkValueInput, 0, pastWorkValueInput->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(pastWorkValueInputBD);
        const algorithmFPType * pastWorkValueInputArray = pastWorkValueInputBD.get();
        result = daal::services::internal::daal_memcpy_s(prevWorkValue.get(), argumentSize * sizeof(algorithmFPType), pastWorkValueInputArray,
                                                         argumentSize * sizeof(algorithmFPType));
    }
    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}

} // namespace internal
} // namespace sgd
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
