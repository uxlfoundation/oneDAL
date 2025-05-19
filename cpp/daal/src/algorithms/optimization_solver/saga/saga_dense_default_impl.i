/* file: saga_dense_default_impl.i */
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
//  Implementation of saga algorithm
//--
*/

#ifndef __SAGA_DENSE_DEFAULT_IMPL_I__
#define __SAGA_DENSE_DEFAULT_IMPL_I__

#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "algorithms/optimization_solver/saga/saga_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace saga
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::optimization_solver::iterative_solver::internal;

/**
 *  \Kernel for Saga calculation
 */

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status SagaKernel<algorithmFPType, method, cpu>::compute(HostAppIface * pHost, NumericTable * inputArgument, NumericTable * minimum,
                                                                   NumericTable * nIterations, NumericTable * gradientsTableInput,
                                                                   NumericTable * gradientsTableResult, Parameter * parameter,
                                                                   engines::BatchBase & engine)
{
    services::Status s;
    int result                = 0;
    const size_t sizeArgument = inputArgument->getNumberOfRows();

    WriteRows<algorithmFPType, cpu> workValueBD(*minimum, 0, sizeArgument);
    DAAL_CHECK_BLOCK_STATUS(workValueBD);
    algorithmFPType * workValue = workValueBD.get();

    ReadRows<algorithmFPType, cpu> initialPointBD(*inputArgument, 0, sizeArgument);
    DAAL_CHECK_BLOCK_STATUS(initialPointBD);
    const algorithmFPType * initialPoint = initialPointBD.get();

    result = daal::services::internal::daal_memcpy_s(workValue, sizeArgument * sizeof(algorithmFPType), initialPoint,
                                                     sizeArgument * sizeof(algorithmFPType));
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

    const size_t maxIterations      = parameter->nIterations;
    const algorithmFPType tolerance = parameter->accuracyThreshold;
    const size_t batchSize          = 1; /* curently only batchSize = 1 supported */

    sum_of_functions::BatchPtr function = parameter->function;
    NumericTablePtr argumentTable(new HomogenNumericTableCPU<algorithmFPType, cpu>(workValue, 1, sizeArgument, s));
    function->sumOfFunctionsInput->set(sum_of_functions::argument, argumentTable);
    function->sumOfFunctionsParameter->resultsToCompute = optimization_solver::objective_function::gradient;

    const size_t n                  = function->sumOfFunctionsParameter->numberOfTerms;
    const algorithmFPType inverse_n = algorithmFPType(1) / algorithmFPType(n);

    sum_of_functions::BatchPtr get_auto_step                 = function->clone();
    get_auto_step->sumOfFunctionsParameter->resultsToCompute = optimization_solver::objective_function::lipschitzConstant;

    NumericTablePtr ntlearningRate = parameter->learningRateSequence;
    ReadRows<algorithmFPType, cpu> learningRateBD;

    algorithmFPType auto_step = 0;
    if (ntlearningRate)
    {
        learningRateBD.set(*ntlearningRate, 0, ntlearningRate->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(learningRateBD);
    }
    else
    {
        get_auto_step->sumOfFunctionsParameter->batchIndices = HomogenNumericTableCPU<int, cpu>::create(nullptr, n, 1);
        get_auto_step->computeNoThrow();
        NumericTablePtr lipschitzPtr = get_auto_step->getResult()->get(optimization_solver::objective_function::lipschitzConstantIdx);
        if (lipschitzPtr)
        {
            ReadRows<algorithmFPType, cpu> lipschitz(*lipschitzPtr, 0, 1);
            DAAL_CHECK_BLOCK_STATUS(lipschitz);
            auto_step = algorithmFPType(1) / (*lipschitz.get());
        }
        else
        {
            auto_step = 0.0001; //default value
        }
    }

    const algorithmFPType * learningRateArray = ntlearningRate ? learningRateBD.get() : nullptr;
    const size_t learningRateLength           = ntlearningRate ? ntlearningRate->getNumberOfColumns() : 0;

    size_t iterationsPerformed = 0;

    RngTask<int, cpu> rngTask(nullptr, 1024);
    DAAL_CHECK_MALLOC(rngTask.init(parameter->function->sumOfFunctionsParameter->numberOfTerms, engine));
    const int * pValues = nullptr;

    algorithmFPType * savedGradients;
    TArray<algorithmFPType, cpu> savedGradientsPtr;
    WriteRows<algorithmFPType, cpu> gradientsTableInputPtr;
    WriteRows<algorithmFPType, cpu> gradientsTableResultPtr;
    if (!gradientsTableInput && !gradientsTableResult)
    {
        savedGradientsPtr.reset(n * sizeArgument);
        savedGradients = savedGradientsPtr.get();
    }
    else
    {
        if (gradientsTableResult)
        {
            gradientsTableResultPtr.set(*gradientsTableResult, 0, n);
            savedGradients = gradientsTableResultPtr.get();
        }
        else
        {
            gradientsTableInputPtr.set(*gradientsTableInput, 0, n);
            savedGradients = gradientsTableInputPtr.get();
        }
    }
    TArray<int, cpu> batchIndicesT(batchSize);
    int * batchIndicesPtr = batchIndicesT.get();

    function->sumOfFunctionsParameter->batchIndices = HomogenNumericTableCPU<int, cpu>::create(batchIndicesPtr, batchSize, 1);

    sum_of_functions::BatchPtr proximaProjection                 = function->clone();
    proximaProjection->sumOfFunctionsParameter->resultsToCompute = optimization_solver::objective_function::proximalProjection;

    TArray<algorithmFPType, cpu> summGradsPtr(sizeArgument);
    algorithmFPType * summGrads = summGradsPtr.get();

    const algorithmFPType * gradient;
    const algorithmFPType * prox;

    function->enableChecks(false);
    proximaProjection->enableChecks(false);

    ReadRows<algorithmFPType, cpu> gradientPtr;
    ReadRows<algorithmFPType, cpu> proxPtr;
    HomogenNumericTable<algorithmFPType> * hmgGradient;
    HomogenNumericTable<algorithmFPType> * hmgProx;
    NumericTablePtr gradientResultPtr;
    NumericTablePtr proxResultPtr;

    /* compute table of gradients if gradientsTableInput is absent */
    if (!gradientsTableInput)
    {
        for (size_t k = 0; k < n; k++)
        {
            batchIndicesPtr[0] = k;
            function->computeNoThrow();
            if (k == 0)
            {
                gradientResultPtr = function->getResult()->get(optimization_solver::objective_function::gradientIdx);

                hmgGradient = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(gradientResultPtr.get());

                gradientPtr.set(*gradientResultPtr, 0, sizeArgument);
                DAAL_CHECK_BLOCK_STATUS(gradientPtr);
                gradient = gradientPtr.get();
            }
            else
            {
                if (!hmgGradient)
                {
                    gradientPtr.set(*gradientResultPtr, 0, sizeArgument);
                    DAAL_CHECK_BLOCK_STATUS(gradientPtr);
                    gradient = gradientPtr.get();
                }
            }

            for (size_t i = 0; i < sizeArgument; i++)
            {
                savedGradients[k * sizeArgument + i] = gradient[i];
            }
        }
    }
    /* compute sum of gradients */
    for (size_t i = 0; i < sizeArgument; i++) summGrads[i] = 0;

    for (size_t k = 0; k < n; k++)
    {
        for (size_t i = 0; i < sizeArgument; i++)
        {
            summGrads[i] += savedGradients[k * sizeArgument + i];
        }
    }

    services::internal::HostAppHelper host(pHost, 10);
    WriteRows<algorithmFPType, cpu> nIterationsPerformed(*nIterations, 0, 1);

    TArray<algorithmFPType, cpu> previousPtr(sizeArgument);
    algorithmFPType * previous = previousPtr.get();

    algorithmFPType stepLength = auto_step;

    if (!s || host.isCancelled(s, 1))
    {
        *nIterationsPerformed.get() = iterationsPerformed;
        return s;
    }

    NumericTablePtr batchIndicesNT = parameter->batchIndices;
    ReadRows<algorithmFPType, cpu> batchIndicesBD;

    if (batchIndicesNT)
    {
        batchIndicesBD.set(*batchIndicesNT, 0, batchIndicesNT->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(batchIndicesBD);
    }

    for (size_t iter = 0; iter < maxIterations; iter++)
    {
        if (batchIndicesNT)
        {
            batchIndicesPtr[0] = batchIndicesBD.get()[iter];
        }
        else
        {
            if (iter % 1024 == 0) rngTask.getWithReplacement(pValues);
            batchIndicesPtr[0] = pValues[iter % 1024];
        }
        const size_t displacement = batchIndicesPtr[0] * sizeArgument;

        iterationsPerformed = iter;

        if (learningRateLength != 0) stepLength = learningRateArray[iter % learningRateLength];
        const algorithmFPType inverse_stepLength = algorithmFPType(1.0) / stepLength;

        s = function->computeNoThrow();

        if (!s || host.isCancelled(s, 1))
        {
            *nIterationsPerformed.get() = iterationsPerformed;
            return s;
        }

        if (iter == 0 && gradientsTableInput)
        {
            gradientResultPtr = function->getResult()->get(optimization_solver::objective_function::gradientIdx);
            hmgGradient       = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(gradientResultPtr.get());
            gradientPtr.set(*gradientResultPtr, 0, sizeArgument);
            DAAL_CHECK_BLOCK_STATUS(gradientPtr);
            gradient = gradientPtr.get();
        }
        else
        {
            if (!hmgGradient)
            {
                gradientPtr.set(*gradientResultPtr, 0, sizeArgument);
                DAAL_CHECK_BLOCK_STATUS(gradientPtr);
                gradient = gradientPtr.get();
            }
        }

        /* TBD use Parallel for */
        result |= daal::services::internal::daal_memcpy_s(previous, sizeArgument * sizeof(algorithmFPType), workValue,
                                                          sizeArgument * sizeof(algorithmFPType));

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t k = 0; k < sizeArgument; k++)
        {
            summGrads[k] += (gradient[k] - savedGradients[displacement + k]);
            workValue[k] = workValue[k] - stepLength * (gradient[k] - savedGradients[displacement + k] + summGrads[k] * inverse_n);
            workValue[k] *= inverse_stepLength;
        }

        s = proximaProjection->computeNoThrow();

        if (iter == 0)
        {
            proxResultPtr = proximaProjection->getResult()->get(optimization_solver::objective_function::proximalProjectionIdx);
            hmgProx       = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(proxResultPtr.get());
            proxPtr.set(*proxResultPtr, 0, sizeArgument);
            DAAL_CHECK_BLOCK_STATUS(proxPtr);
            prox = proxPtr.get();
        }
        else
        {
            if (!hmgProx)
            {
                proxPtr.set(*proxResultPtr, 0, sizeArgument);
                DAAL_CHECK_BLOCK_STATUS(proxPtr);
                prox = proxPtr.get();
            }
        }

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t k = 0; k < sizeArgument; k++)
        {
            workValue[k] = stepLength * prox[k];
        }

        bool continueCheck = false;
        for (size_t k = 0; k < sizeArgument; k++)
        {
            continueCheck |= (daal::internal::MathInst<algorithmFPType, cpu>::sFabs(previous[k] - workValue[k])
                              >= tolerance
                                     * daal::internal::MathInst<algorithmFPType, cpu>::sMax(
                                         1, daal::internal::MathInst<algorithmFPType, cpu>::sFabs(workValue[k])));
        }

        /* number_of_convergence_checks = 1 at first */
        if (!continueCheck)
        {
            break;
        }

        result |= daal::services::internal::daal_memcpy_s(savedGradients + displacement, sizeArgument * sizeof(algorithmFPType), gradient,
                                                          sizeArgument * sizeof(algorithmFPType));
    }
    gradientPtr.release();
    proxPtr.release();
    *nIterationsPerformed.get() = iterationsPerformed + 1;
    return (!result) ? s : Status(ErrorMemoryCopyFailedInternal);
}

} // namespace internal

} // namespace saga

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
