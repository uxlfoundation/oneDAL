/* file: lbfgs_dense_default_impl.i */
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
//  Implementation of LBFGS algorithm
//--
*/

#ifndef __LBFGS_DENSE_DEFAULT_IMPL__
#define __LBFGS_DENSE_DEFAULT_IMPL__

#include "src/externals/service_blas.h"
#include "src/externals/service_rng.h"
#include "src/services/service_data_utils.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace lbfgs
{
namespace internal
{

constexpr double min_curvature = 1e-8;

template <CpuType cpu>
bool getOptionalInputData(NumericTable * pCorrectionIndexInput, size_t & lastCorrectionIndex, size_t & lastIteration)
{
    if (!pCorrectionIndexInput) return false;
    ReadRows<int, cpu> bd(*pCorrectionIndexInput, 0, 1);
    const int * ptr     = bd.get();
    lastCorrectionIndex = size_t(ptr[0]);
    lastIteration       = size_t(ptr[1]);
    return true;
}

static size_t mod(size_t a, size_t m)
{
    return (a - (a / m) * m);
}

/**
 * \brief Kernel for LBFGS calculation
 */
template <typename algorithmFPType, CpuType cpu>
services::Status LBFGSKernel<algorithmFPType, defaultDense, cpu>::compute(
    HostAppIface * pHost, NumericTable * correctionPairsInput, NumericTable * correctionIndicesInput, NumericTable * inputArgument,
    NumericTable * averageArgLIterInput, OptionalArgument * optionalArgumentInput, NumericTable * correctionPairsResult,
    NumericTable * correctionIndicesResult, NumericTable * minimum, NumericTable * nIterationsNT, NumericTable * averageArgLIterResult,
    OptionalArgument * optionalArgumentResult, Parameter * parameter, engines::BatchBase & engine)
{
    services::Status s;
    size_t maxEpoch = parameter->nIterations;
    if (maxEpoch == 0)
    {
        /* Initialize the resulting argument of objective function with the input argument */
        LBFGSTaskBase<algorithmFPType, cpu> task(minimum);
        /* Initialize work value with a start value provided by user */
        DAAL_CHECK_STATUS(s, task.setStartArgument(inputArgument));
        return task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, 0, 0, 0);
    }
    const size_t L                 = parameter->L;
    const size_t m                 = parameter->m;
    const double accuracyThreshold = parameter->accuracyThreshold;

    sum_of_functions::BatchPtr gradientFunction = parameter->function;
    const size_t nTerms                         = gradientFunction->sumOfFunctionsParameter->numberOfTerms;

    const size_t batchSize               = ((parameter->batchSize < nTerms) ? parameter->batchSize : nTerms);
    const size_t correctionPairBatchSize = ((parameter->correctionPairBatchSize < nTerms) ? parameter->correctionPairBatchSize : nTerms);
    const bool useWolfeConditions        = (batchSize == nTerms && correctionPairBatchSize == nTerms && L == 1);
    LBFGSTask<algorithmFPType, cpu> task(parameter, minimum);
    DAAL_CHECK_STATUS(s, task.init(inputArgument, correctionPairsInput, averageArgLIterInput, optionalArgumentInput, parameter, minimum,
                                   averageArgLIterResult, correctionPairsResult, nTerms, batchSize, correctionPairBatchSize));

    NumericTablePtr argumentTable(new HomogenNumericTableCPU<algorithmFPType, cpu>(task.argument, 1, task.argumentSize, s));
    gradientFunction->sumOfFunctionsParameter->batchIndices     = task.ntBatchIndices;
    gradientFunction->sumOfFunctionsParameter->resultsToCompute = objective_function::gradient;

    if (useWolfeConditions)
    {
        gradientFunction->sumOfFunctionsParameter->resultsToCompute |= static_cast<DAAL_UINT64>(objective_function::value);
    }
    gradientFunction->sumOfFunctionsInput->set(sum_of_functions::argument, argumentTable);

    sum_of_functions::BatchPtr hessianFunction                 = gradientFunction->clone();
    hessianFunction->sumOfFunctionsParameter->batchIndices     = task.ntCorrectionPairBatchIndices;
    hessianFunction->sumOfFunctionsParameter->resultsToCompute = objective_function::hessian;
    hessianFunction->sumOfFunctionsInput->set(sum_of_functions::argument, task.argumentLCurTable);

    NumericTablePtr ntGradient;
    NumericTablePtr ntHessian;
    NumericTablePtr ntValue;

    const algorithmFPType invL(algorithmFPType(1.0) / algorithmFPType(L));

    algorithmFPType * argument      = task.argument;
    algorithmFPType * argumentLCur  = task.argumentLCur;
    algorithmFPType * argumentLPrev = task.argumentLPrev;

    size_t lastCorrectionIndex = 0;
    size_t lastIteration       = 0;
    size_t t                   = 0;
    size_t epoch               = 0;
    size_t correctionIndex     = m - 1;
    if (getOptionalInputData<cpu>(correctionIndicesInput, lastCorrectionIndex, lastIteration))
    {
        epoch = lastIteration + 1;
        t     = epoch / L;
        maxEpoch += epoch;
        correctionIndex = lastCorrectionIndex;
    }
    size_t nLIterations                    = maxEpoch / L;
    size_t curIteration                    = 0;
    size_t iPredefinedIndicesCorrectionRow = 0;

    daal::algorithms::engines::internal::BatchBaseImpl * engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl *>(&engine);

    services::internal::HostAppHelper host(pHost, 10);
    for (; t < nLIterations;)
    {
        for (; epoch < (t + 1) * L; ++epoch, ++curIteration)
        {
            for (size_t j = 0; j < task.argumentSize; j++)
            {
                argumentLCur[j] += argument[j];
            }

            bool bContinue = true;
            DAAL_CHECK_STATUS(s, task.updateArgument(curIteration, t, epoch, m, correctionIndex, nTerms, batchSize, accuracyThreshold,
                                                     gradientFunction, ntGradient, ntValue, argument, bContinue, engineImpl, useWolfeConditions));
            if (!bContinue)
            {
                s |= task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, curIteration, epoch, correctionIndex);
                return s;
            }
        }

        for (size_t j = 0; j < task.argumentSize; j++)
        {
            argumentLCur[j] *= invL;
        }

        t++;
        if (t >= 2)
        {
            /* Compute new correction pair */
            correctionIndex = mod(correctionIndex + 1, m);

            if (!useWolfeConditions)
            {
                DAAL_CHECK_STATUS(
                    s, task.updateCorrectionPairBatchIndices(iPredefinedIndicesCorrectionRow, nTerms, correctionPairBatchSize, engineImpl));
            }
            iPredefinedIndicesCorrectionRow++;
            if (!useWolfeConditions)
            {
                s = hessianFunction->computeNoThrow();
            }
            if (!s || host.isCancelled(s, 1))
            {
                s |= task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, curIteration, epoch, correctionIndex);
                return s;
            }
            if (task.correctionPairBatchIndicesStatus == user)
            {
                task.mtCorrectionPairBatchIndices.release();
            }

            if (!useWolfeConditions)
            {
                ntHessian = hessianFunction->getResult()->get(objective_function::hessianIdx);
                DAAL_CHECK_STATUS(s, task.computeCorrectionPair(correctionIndex, ntHessian.get(), useWolfeConditions));
            }
            else
            {
                DAAL_CHECK_STATUS(s, task.computeCorrectionPair(correctionIndex, ntHessian.get(), useWolfeConditions));
            }
        }
        for (size_t j = 0; j < task.argumentSize; j++)
        {
            argumentLPrev[j] = argumentLCur[j];
            argumentLCur[j]  = 0.0;
        }
    }

    for (; epoch < maxEpoch; ++epoch, ++curIteration)
    {
        bool bContinue = true;
        s = task.updateArgument(curIteration, t, epoch, m, correctionIndex, nTerms, batchSize, accuracyThreshold, gradientFunction, ntGradient,
                                ntValue, argument, bContinue, engineImpl, useWolfeConditions);
        if (!s || !bContinue)
        {
            s |= task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, curIteration, epoch, correctionIndex);
            return s;
        }
    }
    return task.setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, curIteration, epoch - 1, correctionIndex);
}

/**
 * Computes the dot product of two vectors
 *
 * \param[in] n     Number of elements in each input vector
 * \param[in] x     Array that contains elements of the first input vector
 * \param[in] y     Array that contains elements of the second input vector
 * \return Resulting dot product
 */
template <typename algorithmFPType, CpuType cpu>
algorithmFPType dotProduct(size_t n, const algorithmFPType * x, const algorithmFPType * y)
{
    algorithmFPType dot = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        dot += x[i] * y[i];
    }
    return dot;
}

/**
 * Line-search
 */
template <typename algorithmFPType, CpuType cpu>
algorithmFPType LBFGSTask<algorithmFPType, cpu>::lineSearch(algorithmFPType * x, NumericTablePtr & ntValue, NumericTablePtr & ntGradient,
                                                            algorithmFPType * dx, sum_of_functions::BatchPtr & gradientFunction,
                                                            bool & continueSearch)
{
    continueSearch                        = false;
    algorithmFPType stepLength            = 1.0;
    const algorithmFPType stepLengthZero  = 0.5;
    const algorithmFPType stepLengthScale = 0.4;

    const algorithmFPType c1            = 0.0001; // Nocendal recomendations
    const algorithmFPType c2            = 0.9;
    const algorithmFPType minStepLength = 0.0000001; // TBD introduce max nIterations
    size_t it                           = 0;
    const DAAL_INT one                  = 1;
    const size_t n                      = this->argumentSize;
    Status s;

    TArray<algorithmFPType, cpu> dnPtr(n);
    algorithmFPType * dn = (algorithmFPType *)dnPtr.get();
    DAAL_CHECK_MALLOC(dn);
    //have to save dx as after computing function it will be changed
    for (size_t i = 0; i < n; i++)
    {
        dn[i] = dx[i];
    }

    ReadRows<algorithmFPType, cpu> fg(*ntGradient, 0, n);
    DAAL_CHECK_BLOCK_STATUS(fg);
    ReadRows<algorithmFPType, cpu> fv(*ntValue, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(fv);

    algorithmFPType term1 = dotProduct<algorithmFPType, cpu>(n, fg.get(), dn);
    algorithmFPType oldFv = fv.get()[0];

    NumericTablePtr ntNewGradient;
    NumericTablePtr ntNewValue;

    while (daal::internal::MathInst<algorithmFPType, cpu>::sFabs(stepLength) > minStepLength)
    {
        // need update x = x + dn*stepLength, where dn = [hessian^(-1)]*gradient
        DAAL_INT nCasted = static_cast<DAAL_INT>(n);
        BlasInst<algorithmFPType, cpu>::xxaxpy(&nCasted, &stepLength, dn, &one, x, &one);

        DAAL_CHECK_STATUS(s, gradientFunction->computeNoThrow());

        ntNewGradient = gradientFunction->getResult()->get(objective_function::gradientIdx);
        ntNewValue    = gradientFunction->getResult()->get(objective_function::valueIdx);
        ReadRows<algorithmFPType, cpu> fgNew(*ntNewGradient, 0, n);
        DAAL_CHECK_BLOCK_STATUS(fgNew);
        ReadRows<algorithmFPType, cpu> fvNew(*ntNewValue, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(fvNew);

        // need to retern initial x (argument)

        const algorithmFPType dnScaleFactor = -stepLength;
        BlasInst<algorithmFPType, cpu>::xxaxpy(&nCasted, &dnScaleFactor, dn, &one, x, &one);

        //Wolfe conditions
        if (fvNew.get()[0] - oldFv <= c1 * stepLength * term1)
        {
            algorithmFPType fgNewXdn = dotProduct<algorithmFPType, cpu>(n, fgNew.get(), dn);
            if (daal::internal::MathInst<algorithmFPType, cpu>::sFabs(fgNewXdn) <= c2 * daal::internal::MathInst<algorithmFPType, cpu>::sFabs(term1))
            {
                break;
            }
        }
        it += 1;
        continueSearch = true;
        stepLength *= (stepLengthZero + stepLengthScale / (algorithmFPType)it); // 0.9, 0.7, 0.63, 0.6, ...
    }

    // retern to initial values
    for (size_t i = 0; i < n; i++)
    {
        dx[i] = dn[i];
    }
    return stepLength;
}
/**
 * Updates argument of the objective function
 *
 * \param[in] t                     Index of the outer loop of LBFGS algorithm
 * \param[in] epoch                 Index of the inner loop of LBFGS algorithm
 * \param[in] m                     Memory parameter of LBFGS
 * \param[in] correctionIndex       Index of starting correction pair in a cyclic buffer
 * \param[in] batchSize             Number of terms of objective function to be used in stochastic gradient
 *                                  computation
 * \param[in] nTerms                Full number of summands (terms) in objective function
 * \param[in] accuracyThreshold     Accuracy of the LBFGS algorithm
 * \param[in] gradientFunction      Objective function for stochastic gradient computations
 * \param[in] ntGradient            Numeric table that stores the stochastic gradient
 * \param[in,out] argument          Array that contains the argument of objective function
 * \param[out] bContinue            Flag. True if the accuracy threshold is not achieved after the update. False otherwise.
 * \param[in] engine                Pointer to RNG engine
 *
 * \return Status of the argument update.
 */
template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::updateArgument(size_t iIteration, size_t t, size_t epoch, size_t m, size_t correctionIndex, size_t nTerms,
                                                       size_t batchSize, algorithmFPType accuracyThreshold,
                                                       sum_of_functions::BatchPtr & gradientFunction, NumericTablePtr & ntGradient,
                                                       NumericTablePtr & ntValue, algorithmFPType * argument, bool & bContinue,
                                                       daal::algorithms::engines::internal::BatchBaseImpl * engine, bool useWolfeConditions)
{
    const algorithmFPType one = 1.0;
    bContinue                 = true;

    Status s;
    int result = 0;

    algorithmFPType * gradientPrev = (algorithmFPType *)_gradientPrevPtr.get();
    algorithmFPType * gradientCurr = (algorithmFPType *)_gradientCurrPtr.get();
    if (!useWolfeConditions)
    {
        DAAL_CHECK_STATUS(
            s, updateBatchIndices(iIteration, nTerms, batchSize, batchIndices, batchIndicesStatus, mtBatchIndices, ntBatchIndices, engine));
    }
    else
    {
        result |= daal::services::internal::daal_memcpy_s(gradientPrev, this->argumentSize * sizeof(algorithmFPType), gradientCurr,
                                                          this->argumentSize * sizeof(algorithmFPType));
        DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
    }
    DAAL_CHECK_STATUS(s, gradientFunction->compute());
    if (batchIndicesStatus == user)
    {
        mtBatchIndices.release();
    }

    ntGradient = gradientFunction->getResult()->get(objective_function::gradientIdx);
    if (useWolfeConditions)
    {
        ntValue = gradientFunction->getResult()->get(objective_function::valueIdx);
    }
    mtGradient.set(*ntGradient, 0, this->argumentSize);
    DAAL_CHECK_BLOCK_STATUS(mtGradient);
    algorithmFPType * gradient = mtGradient.get();
    if (useWolfeConditions)
    {
        result |= daal::services::internal::daal_memcpy_s(gradientCurr, this->argumentSize * sizeof(algorithmFPType), gradient,
                                                          this->argumentSize * sizeof(algorithmFPType));
        DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
    }
    /* Check accuracy */
    if (dotProduct<algorithmFPType, cpu>(this->argumentSize, gradient, gradient)
        < accuracyThreshold
              * daal::internal::MathInst<algorithmFPType, cpu>::sMax(one, dotProduct<algorithmFPType, cpu>(this->argumentSize, argument, argument)))
    {
        mtGradient.release();
        bContinue = false;
        return s;
    }

    /* Get step length on this iteration */
    algorithmFPType stepLengthVal = 0;
    if (t >= 2)
    {
        /* Compute H * gradient */
        twoLoopRecursion(m, correctionIndex, gradient);
    }
    if (useWolfeConditions)
    {
        for (size_t j = 0; j < this->argumentSize; j++)
        {
            gradient[j] *= -1;
        }
        algorithmFPType alpha = 1;
        if (this->continueLineSearch)
        {
            bool continueSearch      = true;
            alpha                    = lineSearch(argument, ntValue, ntGradient, gradient, gradientFunction, continueSearch);
            this->continueLineSearch = continueSearch;
        }
        stepLengthVal = -alpha;
    }
    else
    {
        stepLengthVal = ((nStepLength > 1) ? stepLength[epoch] : stepLength[0]);
    }

    /* Update argument */
    for (size_t j = 0; j < this->argumentSize; j++)
    {
        argument[j] -= stepLengthVal * gradient[j];
    }

    mtGradient.release();
    return s;
}

/**
 * Updates the array of objective function terms indices that are used in stochastic gradient
 * or Hessian matrix computations
 *
 * \param[in]  iPredefinedIndicesRow    Index of row of predefined indices to use
 * \param[in]  nTerms                   Full number of summands (terms) in objective function
 * \param[in]  batchSize                Number of terms of objective function to be used in stochastic gradient
 *                                      or Hessian matrix computations
 * \param[out] batchIndices             Array of indices of objective function terms that are used
 *                                      in stochastic gradient or Hessian matrix computations
 * \param[in]  batchIndicesStatus       Status of the indices of objective function terms
 * \param[in]  mtBatchIndices           Block of rows of the numeric table that stores the indices of objective function terms
 * \param[in]  ntBatchIndices           Numeric table that stores the indices of objective function terms
 * \param[in]  engine                   Pointer to RNG engine
 */
template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::updateBatchIndices(size_t iPredefinedIndicesRow, size_t nTerms, size_t batchSize, int *& batchIndices,
                                                           IndicesStatus batchIndicesStatus, daal::internal::ReadRows<int, cpu> & mtBatchIndices,
                                                           services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > & ntBatchIndices,
                                                           daal::algorithms::engines::internal::BatchBaseImpl * engine)
{
    if (batchIndicesStatus == all)
    {
        return Status();
    }
    else if (batchIndicesStatus == user)
    {
        mtBatchIndices.next(iPredefinedIndicesRow, 1);
        DAAL_CHECK_BLOCK_STATUS(mtBatchIndices);
        batchIndices = const_cast<int *>(mtBatchIndices.get());
        ntBatchIndices->setArray(batchIndices, ntBatchIndices->getNumberOfRows());
    }
    else // if (batchIndicesStatus == random)
    {
        DAAL_CHECK(nTerms <= services::internal::MaxVal<int>::get(), ErrorIncorrectNumberOfTerms)
        DAAL_CHECK(_rng.uniformWithoutReplacement(batchSize, batchIndices, engine->getState(), 0, (int)nTerms) == 0,
                   ErrorIncorrectErrorcodeFromGenerator);
    }
    return Status();
}

/**
 * Two-loop recursion algorithm that computes approximation of inverse Hessian matrix
 * multiplied by input gradient vector from a set of correction pairs (s(j), y(j)), j = 1,...,m.
 *
 * See Algorithm 7.4 in [2].
 *
 * \param[in]  m               Number of correction pairs
 * \param[in]  correctionIndex Index of starting correction pair in a cyclic buffer
 * \param[in,out] gradient     On input:  Gradient vector.
 *                             On output: iterative_solver::Result of two-loop recursion.
 */
template <typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::twoLoopRecursion(size_t m, size_t correctionIndex, algorithmFPType * gradient)
{
    for (size_t k = 0; k < m; k++)
    {
        const size_t index                     = mod(correctionIndex + m - 1 - k, m);
        const algorithmFPType * correctionSPtr = correctionS + index * this->argumentSize;
        const algorithmFPType * correctionYPtr = correctionY + index * this->argumentSize;

        alpha[index] = rho[index] * dotProduct<algorithmFPType, cpu>(this->argumentSize, correctionSPtr, gradient);

        for (size_t j = 0; j < this->argumentSize; j++)
        {
            gradient[j] -= alpha[index] * correctionYPtr[j];
        }
    }

    for (size_t k = 0; k < m; k++)
    {
        const size_t index                     = mod(correctionIndex + k, m);
        const algorithmFPType * correctionSPtr = correctionS + index * this->argumentSize;
        const algorithmFPType * correctionYPtr = correctionY + index * this->argumentSize;

        algorithmFPType beta = rho[index] * dotProduct<algorithmFPType, cpu>(this->argumentSize, correctionYPtr, gradient);

        for (size_t j = 0; j < this->argumentSize; j++)
        {
            gradient[j] += correctionSPtr[j] * (alpha[index] - beta);
        }
    }
}

/**
* Computes the correction pair (s, y) and the corresponding value rho
*
* \param[out] correctionIndex  Index of correction pair to be changed
* \param[in]  hessian          Hessian numeric table
*/
template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::computeCorrectionPair(size_t correctionIndex, NumericTable * ntHessian, bool useWolfeConditions)
{
    if (useWolfeConditions)
    {
        computeCorrectionPairImpl(correctionIndex, NULL, useWolfeConditions);
    }
    else
    {
        mtHessian.set(*ntHessian, 0, this->argumentSize);
        DAAL_CHECK_BLOCK_STATUS(mtHessian);
        computeCorrectionPairImpl(correctionIndex, mtHessian.get(), useWolfeConditions);
        mtHessian.release();
    }
    return Status();
}

/**
 * Computes the correction pair (s, y) and the corresponding value rho
 *
 * \param[out] correctionIndex  Index of correction pair to be changed
 * \param[in]  hessian          Approximation of Hessian matrix of the objective function on the current iteration
 *                              See formula (7.17) in [2]
 */
template <typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::computeCorrectionPairImpl(size_t correctionIndex, const algorithmFPType * hessian, bool useWolfeConditions)
{
    algorithmFPType * s = correctionS + correctionIndex * this->argumentSize;
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t j = 0; j < this->argumentSize; j++)
    {
        s[j] = argumentLCur[j] - argumentLPrev[j];
    }

    algorithmFPType * y  = correctionY + correctionIndex * this->argumentSize;
    char trans           = 'N';
    algorithmFPType zero = 0.0;
    algorithmFPType one  = 1.0;
    DAAL_INT n           = (DAAL_INT)(this->argumentSize);
    DAAL_INT ione        = 1;
    if (useWolfeConditions)
    {
        algorithmFPType * gradientPrev = (algorithmFPType *)_gradientPrevPtr.get();
        algorithmFPType * gradientCurr = (algorithmFPType *)_gradientCurrPtr.get();

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < this->argumentSize; j++)
        {
            y[j] = (gradientCurr[j] - gradientPrev[j]);
        }
    }
    else
    {
        BlasInst<algorithmFPType, cpu>::xgemv(&trans, &n, &n, &one, const_cast<algorithmFPType *>(hessian), &n, s, &ione, &zero, y, &ione);
    }
    rho[correctionIndex] = dotProduct<algorithmFPType, cpu>(this->argumentSize, s, y);
    if (rho[correctionIndex] >= min_curvature) // threshold
    {
        rho[correctionIndex] = 1.0 / rho[correctionIndex];
    }
}

/**
 * Creates structure for storing data used in itermediate computations in LBFGS algorithm
 * if number of iterations is 0
 *
 * \param[in] argumentTable     Numeric table that stores the argument of objective function
 */
template <typename algorithmFPType, CpuType cpu>
LBFGSTaskBase<algorithmFPType, cpu>::LBFGSTaskBase(NumericTable * argumentTable)
    : mtArgument(argumentTable, 0, argumentTable->getNumberOfRows()), argument(nullptr), argumentSize(argumentTable->getNumberOfRows())
{}

/**
* Sets the initial argument of objective function
*
* \param[in] startValueTable   Numeric table that stores the starting point, the initial argument of objective function
*/
template <typename algorithmFPType, CpuType cpu>
Status LBFGSTaskBase<algorithmFPType, cpu>::setStartArgument(NumericTable * startValueTable)
{
    /* Initialize work value with a start value provided by user */
    DAAL_CHECK_BLOCK_STATUS(mtArgument);
    argument = mtArgument.get();
    ReadRows<algorithmFPType, cpu> mtStartValue(startValueTable, 0, argumentSize);
    DAAL_CHECK_BLOCK_STATUS(mtStartValue);
    int result = daal::services::internal::daal_memcpy_s(argument, argumentSize * sizeof(algorithmFPType), mtStartValue.get(),
                                                         argumentSize * sizeof(algorithmFPType));
    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
Status LBFGSTaskBase<algorithmFPType, cpu>::setToResult(NumericTable * correctionIndicesResult, NumericTable * nIterationsNT,
                                                        OptionalArgument * optionalArgumentResult, size_t nIterations, size_t epoch, size_t corrIndex)
{
    WriteRows<int, cpu> mtNIterations(nIterationsNT, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(mtNIterations);
    DAAL_CHECK(nIterations <= services::internal::MaxVal<int>::get(), ErrorIterativeSolverIncorrectMaxNumberOfIterations)
    *mtNIterations.get() = (int)nIterations;

    if (correctionIndicesResult)
    {
        WriteRows<int, cpu> bd(*correctionIndicesResult, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(bd);
        int * ptr = bd.get();
        ptr[0]    = corrIndex;
        ptr[1]    = epoch;
    }
    return Status();
}

/**
* Creates structure for storing data used in itermediate computations in LBFGS algorithm
*
* \param[in] parameter         Algorithm's parameter
* \param[in] minimum           Algorithm's result
*/
template <typename algorithmFPType, CpuType cpu>
LBFGSTask<algorithmFPType, cpu>::LBFGSTask(const Parameter * parameter, NumericTable * minimum)
    : super(minimum),
      mtBatchIndices(parameter->batchIndices.get()),
      mtCorrectionPairBatchIndices(parameter->correctionPairBatchIndices.get()),
      mtStepLength(parameter->stepLengthSequence.get(), 0, 1),
      correctionPairs(nullptr),
      correctionS(nullptr),
      correctionY(nullptr),
      batchIndices(nullptr),
      batchIndicesStatus(all),
      correctionPairBatchIndices(nullptr),
      correctionPairBatchIndicesStatus(all),
      nStepLength(parameter->stepLengthSequence->getNumberOfColumns()),
      _rng()
{}

/**
* Initializes the task for itermediate computations in LBFGS algorithm
*
* \param[in] input             Algorithm's input
* \param[in] parameter         Algorithm's parameter
* \param[in] result            Algorithm's result
* \param[in] nTerms            Full number of summands (terms) in objective function
* \param[in] batchSize         Number of terms to compute the stochastic gradient
* \param[in] correctionPairBatchSize           Number of terms to compute the sub-sampled Hessian
*                                              for correction pairs computation
*/
template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::init(NumericTable * inputArgument, NumericTable * correctionPairsInput, NumericTable * averageArgLIterInput,
                                             OptionalArgument * optionalArgumentInput, const Parameter * parameter, NumericTable * minimum,
                                             NumericTable * averageArgLIterResult, NumericTable * correctionPairsResult, size_t nTerms,
                                             size_t batchSize, size_t correctionPairBatchSize)
{
    Status s;
    this->continueLineSearch = true;
    /* Initialize work value with a start value provided by user */
    DAAL_CHECK_STATUS(s, this->setStartArgument(inputArgument));
    DAAL_CHECK_STATUS(s, initArgumentL(averageArgLIterInput, averageArgLIterResult, parameter));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, parameter->m, sizeof(algorithmFPType));
    alpha = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(parameter->m * sizeof(algorithmFPType));
    DAAL_CHECK_MALLOC(argumentLCur && argumentLPrev && alpha);

    DAAL_CHECK_STATUS(s, initCorrectionPairs(correctionPairsInput, parameter, correctionPairsResult));

    /* Get step-length sequence */
    DAAL_CHECK_BLOCK_STATUS(mtStepLength);
    stepLength = mtStepLength.get();

    /* Create numeric table for storing the average of work values for the last L iterations */
    argumentLCurTable.reset(new HomogenNumericTableCPU<algorithmFPType, cpu>(argumentLCur, 1, this->argumentSize, s));
    DAAL_CHECK_MALLOC(argumentLCurTable.get());

    /* Initialize indices for objective function gradient computations */
    NumericTable * batchIndicesTable = parameter->batchIndices.get();
    if (batchSize < nTerms)
    {
        DAAL_CHECK_STATUS(s, getBatchIndices(batchSize, batchIndicesTable, batchIndices, batchIndicesStatus, ntBatchIndices));
    }

    /* Initialize indices for objective function Hessian computations */
    if (correctionPairBatchSize < nTerms)
    {
        DAAL_CHECK_STATUS(s, getBatchIndices(correctionPairBatchSize, parameter->correctionPairBatchIndices.get(), correctionPairBatchIndices,
                                             correctionPairBatchIndicesStatus, ntCorrectionPairBatchIndices));
    }

    const bool useWolfeConditions = (batchSize == parameter->function->sumOfFunctionsParameter->numberOfTerms
                                     && correctionPairBatchSize == parameter->function->sumOfFunctionsParameter->numberOfTerms && parameter->L == 1);
    if (useWolfeConditions)
    {
        _gradientPrevPtr.reset(this->argumentSize);
        _gradientCurrPtr.reset(this->argumentSize);
    }

    return s;
}

template <typename algorithmFPType, CpuType cpu>
LBFGSTask<algorithmFPType, cpu>::~LBFGSTask()
{
    if (argumentLCur && !argumentLCurRows.get())
    {
        daal_free(argumentLCur);
        argumentLCur = nullptr;
    }
    if (argumentLPrev && !argumentLPrevRows.get())
    {
        daal_free(argumentLPrev);
        argumentLPrev = nullptr;
    }
    if (rho)
    {
        daal_free(rho);
        rho = nullptr;
    }
    if (alpha)
    {
        daal_free(alpha);
        alpha = nullptr;
    }
    if (correctionPairs)
    {
        correctionPairs->releaseBlockOfRows(correctionPairsBD);
    }
    else
    {
        if (correctionS)
        {
            daal_free(correctionS);
            correctionS = nullptr;
        }
        if (correctionY)
        {
            daal_free(correctionY);
            correctionY = nullptr;
        }
    }
    if (stepLength)
    {
        mtStepLength.release();
    }
    releaseBatchIndices(batchIndices, batchIndicesStatus);
    releaseBatchIndices(correctionPairBatchIndices, correctionPairBatchIndicesStatus);
}

template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::setToResult(NumericTable * correctionIndicesResult, NumericTable * nIterationsNT,
                                                    OptionalArgument * optionalArgumentResult, size_t nIterations, size_t epoch, size_t corrIndex)
{
    Status s;
    DAAL_CHECK_STATUS(s, super::setToResult(correctionIndicesResult, nIterationsNT, optionalArgumentResult, nIterations, epoch, corrIndex));
    return s;
}

template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::initArgumentL(NumericTable * averageArgLIterInput, NumericTable * averageArgLIterResult,
                                                      const Parameter * parameter)
{
    NumericTable * pOptRes = parameter->optionalResultRequired ? averageArgLIterResult : nullptr;
    int result             = 0;
    if (pOptRes)
    {
        argumentLPrevRows.set(pOptRes, 0, 1);
        argumentLPrev = argumentLPrevRows.get();
        argumentLCurRows.set(pOptRes, 1, 1);
        argumentLCur = argumentLCurRows.get();
    }
    else
    {
        argumentLCur  = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(this->argumentSize);
        argumentLPrev = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(this->argumentSize);
        DAAL_CHECK_MALLOC(argumentLCur && argumentLPrev);
    }

    if (averageArgLIterInput)
    {
        if (averageArgLIterInput != pOptRes)
        {
            ReadRows<algorithmFPType, cpu> rr(averageArgLIterInput, 0, 1);
            DAAL_CHECK_BLOCK_STATUS(rr);
            const auto cMemSize = sizeof(algorithmFPType) * averageArgLIterInput->getNumberOfColumns();
            result |= daal::services::internal::daal_memcpy_s(argumentLPrev, cMemSize, rr.get(), cMemSize);
            rr.next(1, 1);
            result |= daal::services::internal::daal_memcpy_s(argumentLCur, cMemSize, rr.get(), cMemSize);
        }
    }
    else if (pOptRes)
    {
        daal::services::internal::service_memset<algorithmFPType, cpu>(argumentLPrev, 0, this->argumentSize);
        daal::services::internal::service_memset<algorithmFPType, cpu>(argumentLCur, 0, this->argumentSize);
    }
    return (!result) ? Status() : Status(ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::initCorrectionPairs(NumericTable * correctionPairsInput, const Parameter * parameter,
                                                            NumericTable * correctionPairsResult)
{
    Status s;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, parameter->m, this->argumentSize);
    const auto cCorrectionPairSize       = parameter->m * this->argumentSize;
    NumericTable * correctionPairsOutput = parameter->optionalResultRequired ? correctionPairsResult : nullptr;
    if (correctionPairsOutput)
    {
        correctionPairs = correctionPairsOutput;
        DAAL_CHECK_STATUS(s, correctionPairs->getBlockOfRows(0, 2 * parameter->m, readWrite, correctionPairsBD));
        correctionS = correctionPairsBD.getBlockPtr();
        if (correctionS) correctionY = correctionS + cCorrectionPairSize; /* second half */
    }
    else
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, cCorrectionPairSize, sizeof(algorithmFPType));
        correctionS = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(cCorrectionPairSize);
        correctionY = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(cCorrectionPairSize);
    }
    rho = (algorithmFPType *)daal::services::internal::service_calloc<algorithmFPType, cpu>(parameter->m);
    DAAL_CHECK_MALLOC(correctionS && correctionY && rho);

    /* If input correction pairs are given ... */
    if (correctionPairsInput)
    {
        /* and if it is not the same as output correction pairs then copy them to correctionS, correctionY */
        if (correctionPairsOutput != correctionPairsInput)
        {
            ReadRows<algorithmFPType, cpu> correctionPairsInputBD(*correctionPairsInput, 0, correctionPairsInput->getNumberOfRows());
            DAAL_CHECK_BLOCK_STATUS(correctionPairsInputBD);
            const auto cMemSize = sizeof(algorithmFPType) * cCorrectionPairSize;
            int result          = 0;
            result |= daal::services::internal::daal_memcpy_s(correctionS, cMemSize, correctionPairsInputBD.get(), cMemSize);
            result |= daal::services::internal::daal_memcpy_s(correctionY, cMemSize, correctionPairsInputBD.get() + cMemSize, cMemSize);
            DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
        }
        /* initialize rho form S and Y */
        for (auto i = 0; i < parameter->m; ++i)
        {
            auto s = correctionS + i * this->argumentSize;
            auto y = correctionY + i * this->argumentSize;
            rho[i] = dotProduct<algorithmFPType, cpu>(this->argumentSize, s, y);
            if (rho[i] >= min_curvature) // threshold
            {
                rho[i] = 1.0 / rho[i];
            }
        }
    }
    else if (correctionPairsOutput)
    {
        daal::services::internal::service_memset<algorithmFPType, cpu>(correctionS, 0, cCorrectionPairSize);
        daal::services::internal::service_memset<algorithmFPType, cpu>(correctionY, 0, cCorrectionPairSize);
    }
    return Status();
}

/**
 * Returns the array that contains the indices of objective function terms used for
 * stochastic gradient or sub-sampled Hessian matrix computation
 *
 * \param[in]  size             Number of indices
 * \param[in]  indicesTable     Numeric table that represent indices that will be used
 *                              instead of random values for the stochastic gradient
 *                              or sub-sampled Hessian matrix computations
 * \param[out] indices          Resulting array that contains the indices provided by user
 *                              or memory for storing randomly generated indices
 * \param[out] indicesStatusPtr Status of the indices array
 * \param[out] ntIndices        Numeric table that stores the indices
 */
template <typename algorithmFPType, CpuType cpu>
Status LBFGSTask<algorithmFPType, cpu>::getBatchIndices(size_t size, NumericTable * indicesTable, int *& indices, IndicesStatus & indicesStatus,
                                                        services::SharedPtr<daal::internal::HomogenNumericTableCPU<int, cpu> > & ntIndices)
{
    Status s;
    if (indicesTable)
    {
        indicesStatus = user;
    }
    else
    {
        indicesStatus = random;
        indices       = (int *)daal::services::internal::service_calloc<int, cpu>(size * sizeof(int));
        DAAL_CHECK_MALLOC(indices);
    }

    ntIndices.reset(new HomogenNumericTableCPU<int, cpu>(indices, size, 1, s));
    DAAL_CHECK_MALLOC(ntIndices.get());
    return s;
}

/**
 * Releases the array that contains the indices of objective function terms used for
 * stochastic gradient or sub-sampled Hessian matrix computation
 *
 * \param[in]  indices       Array that contains the indices provided by user
 *                           or memory for storing randomly generated indices
 * \param[in]  indicesStatus Status of the array of indices
 */
template <typename algorithmFPType, CpuType cpu>
void LBFGSTask<algorithmFPType, cpu>::releaseBatchIndices(int * indices, IndicesStatus indicesStatus)
{
    if (indicesStatus == random)
    {
        daal_free(indices);
        indices = nullptr;
    }
}

} // namespace internal

} // namespace lbfgs

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal

#endif
