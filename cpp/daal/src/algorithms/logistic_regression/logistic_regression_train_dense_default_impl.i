/* file: logistic_regression_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for logistic regression classification
//  (defaultDense) method.
//--
*/

#ifndef __LOGISTIC_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __LOGISTIC_REGRESSION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/logistic_regression/logistic_regression_train_kernel.h"
#include "src/algorithms/logistic_regression/logistic_regression_model_impl.h"
#include "src/algorithms/service_error_handling.h"
#include "src/services/service_algo_utils.h"
#include "algorithms/optimization_solver/objective_function/logistic_loss_batch.h"
#include "algorithms/optimization_solver/objective_function/cross_entropy_loss_batch.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_profiler.h"

using namespace daal::algorithms::logistic_regression::training::internal;
using namespace daal::algorithms::optimization_solver;
using namespace daal;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, logistic_regression::training::Method method, CpuType cpu>
services::Status TrainBatchKernel<algorithmFPType, method, cpu>::compute(const HostAppIfacePtr & pHost, const NumericTablePtr & x,
                                                                         const NumericTablePtr & y, logistic_regression::Model & m, Result & res,
                                                                         const Parameter & par)
{
    const size_t p = x->getNumberOfColumns() + 1;
    DAAL_ASSERT(p == m.getNumberOfBetas());
    services::SharedPtr<optimization_solver::iterative_solver::Batch> pSolver = par.optimizationSolver->clone();
    pSolver->setHostApp(pHost);
    if (par.nClasses == 2)
    {
        services::SharedPtr<logistic_loss::Batch<algorithmFPType> > objFunc(logistic_loss::Batch<algorithmFPType>::create(x->getNumberOfRows()));
        objFunc->input.set(logistic_loss::data, x);
        objFunc->input.set(logistic_loss::dependentVariables, y);
        objFunc->parameter().interceptFlag = par.interceptFlag;
        objFunc->parameter().penaltyL1     = par.penaltyL1;
        objFunc->parameter().penaltyL2     = par.penaltyL2;
        pSolver->getParameter()->function  = objFunc;
    }
    else
    {
        services::SharedPtr<cross_entropy_loss::Batch<algorithmFPType> > objFunc(
            cross_entropy_loss::Batch<algorithmFPType>::create(par.nClasses, x->getNumberOfRows()));
        objFunc->input.set(cross_entropy_loss::data, x);
        objFunc->input.set(cross_entropy_loss::dependentVariables, y);
        objFunc->parameter().interceptFlag = par.interceptFlag;
        objFunc->parameter().penaltyL1     = par.penaltyL1;
        objFunc->parameter().penaltyL2     = par.penaltyL2;
        pSolver->getParameter()->function  = objFunc;
    }

    const size_t nBetaRows  = m.getBeta()->getNumberOfRows();
    const size_t nBetaTotal = p * nBetaRows;
    services::Status s;

    NumericTablePtr pArg = daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(1, nBetaTotal, &s);
    DAAL_CHECK_STATUS_VAR(s);

    if (!s) return s;

    //initialization
    {
        const size_t nRows = y->getNumberOfRows();
        daal::internal::ReadRows<algorithmFPType, cpu> yrows(*y, 0, nRows);
        daal::internal::WriteRows<algorithmFPType, cpu> argRows(*pArg, 0, nBetaTotal);
        DAAL_CHECK_BLOCK_STATUS(yrows);
        DAAL_CHECK_BLOCK_STATUS(argRows);
        const algorithmFPType * py = yrows.get();
        algorithmFPType * pb       = argRows.get();
        daal::services::internal::service_memset<algorithmFPType, cpu>(pb, 0, nBetaTotal);
        if (par.nClasses == 2)
        {
            size_t count = 0;
            for (size_t i = 0; i < nRows; ++i) count += (py[i] != 0);
            algorithmFPType initialVal = 1;
            if (count && (count != nRows))
            {
                auto val   = algorithmFPType(count) / (algorithmFPType(nRows) - algorithmFPType(count)); //mean/(1-mean)
                initialVal = daal::internal::MathInst<algorithmFPType, cpu>::sLog(val);
            }
            pb[0] = initialVal;
        }
        else
        {
            const algorithmFPType initialVal = 1e-3;
            for (size_t i = 0; i < par.nClasses; ++i) pb[i * p + 0] = initialVal;
        }
    }
    //initialize solver arguments
    pSolver->getInput()->set(optimization_solver::iterative_solver::inputArgument, pArg);
    DAAL_CHECK_STATUS(s, pSolver->compute());

    NumericTablePtr nIterationsOut = pSolver->getResult()->get(optimization_solver::iterative_solver::nIterations);

    par.optimizationSolver->getResult()->set(optimization_solver::iterative_solver::nIterations, nIterationsOut);

    //write data to model
    daal::internal::ReadRows<algorithmFPType, cpu> ar(*pSolver->getResult()->get(optimization_solver::iterative_solver::minimum), 0, nBetaTotal);
    daal::internal::WriteRows<algorithmFPType, cpu> br(*m.getBeta(), 0, nBetaRows);
    DAAL_CHECK_BLOCK_STATUS(ar);
    DAAL_CHECK_BLOCK_STATUS(br);
    const algorithmFPType * a = ar.get();
    algorithmFPType * pBeta   = br.get();
    for (size_t j = 0; j < nBetaTotal; j++) pBeta[j] = a[j];

    if (!par.interceptFlag)
    {
        for (size_t j = 0; j < nBetaRows; ++j) pBeta[p * j + 0] = 0;
    }
    return s;
}

} /* namespace internal */
} /* namespace training */
} /* namespace logistic_regression */
} /* namespace algorithms */
} /* namespace daal */

#endif
