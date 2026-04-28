/* file: coordinate_descent_dense_default_impl.i */
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
//  Implementation of coordinate_descent algorithm
//--
*/

#ifndef __COORDINATE_DESCENT_DENSE_DEFAULT_IMPL_I__
#define __COORDINATE_DESCENT_DENSE_DEFAULT_IMPL_I__

#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/services/service_utils.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/optimization_solver/iterative_solver_kernel.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_types.h"
#include "algorithms/optimization_solver/coordinate_descent/coordinate_descent_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace coordinate_descent
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::algorithms::optimization_solver::iterative_solver::internal;

/**
 *  \Kernel for CoordinateDescent calculation
 */

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CoordinateDescentKernel<algorithmFPType, method, cpu>::compute(NumericTable * inputArgument, NumericTable * minimum,
                                                                                NumericTable * nIterations, Parameter * parameter,
                                                                                engines::BatchBase & engine,
                                                                                optimization_solver::objective_function::ResultPtr & hesGrResultPtr,
                                                                                optimization_solver::objective_function::ResultPtr & proxResultPtr)
{
    services::Status s;

    const size_t nRowsArgument              = inputArgument->getNumberOfRows();
    const size_t nColsArgument              = inputArgument->getNumberOfColumns();
    const algorithmFPType accuracyThreshold = parameter->accuracyThreshold;
    WriteRows<algorithmFPType, cpu> workValueBD(*minimum, 0, nRowsArgument);
    DAAL_CHECK_BLOCK_STATUS(workValueBD);
    algorithmFPType * const workValue = workValueBD.get();

    WriteRows<algorithmFPType, cpu> nIterationsBD(*nIterations, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(nIterationsBD);
    algorithmFPType * const nIter = nIterationsBD.get();
    ReadRows<algorithmFPType, cpu> initialPointBD(*inputArgument, 0, nRowsArgument);
    DAAL_CHECK_BLOCK_STATUS(initialPointBD);
    const algorithmFPType * initialPoint = initialPointBD.get();

    int result = daal::services::internal::daal_memcpy_s(workValue, nRowsArgument * nColsArgument * sizeof(algorithmFPType), initialPoint,
                                                         nRowsArgument * nColsArgument * sizeof(algorithmFPType));
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

    sum_of_functions::BatchPtr gradientHessianFunction = parameter->function->clone();
    const size_t maxIterations                         = parameter->nIterations;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nColsArgument, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsArgument, nColsArgument);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsArgument * nColsArgument, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> argumentForProximalT(nRowsArgument * nColsArgument);
    algorithmFPType * const argumentForProximal = argumentForProximalT.get();
    DAAL_CHECK_MALLOC(argumentForProximal);
    sum_of_functions::BatchPtr proximalProjectionFunction = gradientHessianFunction->clone();

    daal::services::internal::service_memset<algorithmFPType, cpu>(argumentForProximal, 0, nRowsArgument * nColsArgument);
    NumericTablePtr argumentForProximalTable =
        HomogenNumericTableCPU<algorithmFPType, cpu>::create(argumentForProximal, nColsArgument, nRowsArgument, &s);

    proximalProjectionFunction->sumOfFunctionsInput->set(sum_of_functions::argument, argumentForProximalTable);
    proximalProjectionFunction->sumOfFunctionsParameter->resultsToCompute = optimization_solver::objective_function::componentOfProximalProjection;

    NumericTablePtr argumentTable = HomogenNumericTableCPU<algorithmFPType, cpu>::create(workValue, nColsArgument, nRowsArgument, &s);

    gradientHessianFunction->sumOfFunctionsInput->set(sum_of_functions::argument, argumentTable);

    gradientHessianFunction->sumOfFunctionsParameter->resultsToCompute =
        optimization_solver::objective_function::componentOfGradient | optimization_solver::objective_function::componentOfHessianDiagonal;

    TArray<algorithmFPType, cpu> grBD(nColsArgument);
    algorithmFPType * const iGr = grBD.get();
    DAAL_CHECK_MALLOC(iGr);
    daal::services::internal::service_memset<algorithmFPType, cpu>(iGr, 0, nColsArgument);

    TArray<algorithmFPType, cpu> hBD(nColsArgument);
    algorithmFPType * const iHes = hBD.get();
    DAAL_CHECK_MALLOC(iHes);
    daal::services::internal::service_memset<algorithmFPType, cpu>(iHes, 0, nColsArgument);

    TArray<algorithmFPType, cpu> pBD(nColsArgument);
    algorithmFPType * const iPr = pBD.get();
    DAAL_CHECK_MALLOC(iPr);
    daal::services::internal::service_memset<algorithmFPType, cpu>(iPr, 0, nColsArgument);

    NumericTablePtr gNt = HomogenNumericTableCPU<algorithmFPType, cpu>::create(iGr, nColsArgument, 1, &s);
    NumericTablePtr hNt = HomogenNumericTableCPU<algorithmFPType, cpu>::create(iHes, nColsArgument, 1, &s);
    NumericTablePtr pNt = HomogenNumericTableCPU<algorithmFPType, cpu>::create(iPr, nColsArgument, 1, &s);

    hesGrResultPtr->set(optimization_solver::objective_function::componentOfGradientIdx, gNt);
    hesGrResultPtr->set(optimization_solver::objective_function::componentOfHessianDiagonalIdx, hNt);
    proxResultPtr->set(optimization_solver::objective_function::componentOfProximalProjectionIdx, pNt);
    proxResultPtr->set(optimization_solver::objective_function::componentOfHessianDiagonalIdx, hNt);

    gradientHessianFunction->setResult(hesGrResultPtr);
    proximalProjectionFunction->setResult(proxResultPtr);

    gradientHessianFunction->enableChecks(false);
    proximalProjectionFunction->enableChecks(false);

    algorithmFPType maxDiff  = 0;
    algorithmFPType maxValue = 0;
    TArray<algorithmFPType, cpu> prewsT(nColsArgument);
    TArray<algorithmFPType, cpu> stepsT(nColsArgument);
    TArray<algorithmFPType, cpu> proxsT(nColsArgument);

    algorithmFPType * const prews = prewsT.get();
    algorithmFPType * const steps = stepsT.get();
    algorithmFPType * const proxs = proxsT.get();
    DAAL_CHECK_MALLOC(prews);
    DAAL_CHECK_MALLOC(steps);
    DAAL_CHECK_MALLOC(proxs);

    const bool positive    = parameter->positive;
    const size_t startedId = parameter->skipTheFirstComponents ? 1 : 0;
    size_t itr             = 0;
    for (itr = 0; itr < maxIterations; itr++)
    {
        for (size_t id = startedId; id < (nRowsArgument); id++)
        {
            //const algorithmFPType prew = workValue[id];
            for (size_t ic = 0; ic < nColsArgument; ic++)
            {
                prews[ic] = workValue[id * nColsArgument + ic];
            }
            gradientHessianFunction->sumOfFunctionsParameter->featureId = id;
            gradientHessianFunction->computeNoThrow();

            for (size_t ic = 0; ic < nColsArgument; ic++)
            {
                steps[ic] = (algorithmFPType)1.0 / (iHes[ic] == 0 ? 1 : iHes[ic]);
            }

            for (size_t ic = 0; ic < nColsArgument; ic++)
            {
                proxs[ic] = prews[ic] - steps[ic] * iGr[ic];
            }
            if (positive)
            {
                for (size_t ic = 0; ic < nColsArgument; ic++)
                {
                    proxs[ic] = proxs[ic] < 0 ? 0 : proxs[ic];
                }
            }

            proximalProjectionFunction->sumOfFunctionsParameter->featureId = id;
            for (size_t ic = 0; ic < nColsArgument; ic++)
            {
                const algorithmFPType inversStep             = (algorithmFPType)1.0 / (steps[ic]);
                argumentForProximal[id * nColsArgument + ic] = inversStep * proxs[ic];
            }
            proximalProjectionFunction->computeNoThrow();

            for (size_t ic = 0; ic < nColsArgument; ic++)
            {
                workValue[id * nColsArgument + ic] = (iHes[ic] == 0) ? workValue[id * nColsArgument + ic] : iPr[ic] * steps[ic];
            }

            for (size_t ic = 0; ic < nColsArgument; ic++)
            {
                const algorithmFPType diff = daal::internal::MathInst<algorithmFPType, cpu>::sFabs(prews[ic] - workValue[id * nColsArgument + ic]);
                const algorithmFPType maxValueCurr = daal::internal::MathInst<algorithmFPType, cpu>::sFabs(workValue[id * nColsArgument + ic]);
                maxDiff                            = diff > maxDiff ? diff : maxDiff;
                maxValue                           = maxValueCurr > maxValue ? maxValueCurr : maxValue;
            }
        }
        if (maxDiff <= accuracyThreshold * maxValue)
        {
            break;
        }
        maxValue = 0;
        maxDiff  = 0;
    }
    *nIter = itr + 1;
    return s;
}

} // namespace internal
} // namespace coordinate_descent
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
