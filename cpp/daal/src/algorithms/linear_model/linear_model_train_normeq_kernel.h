/* file: linear_model_train_normeq_kernel.h */
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
//  Declaration of common base classes for normal equations model training.
//--
*/

#ifndef __LINEAR_MODEL_TRAIN_NORMEQ_KERNEL_H__
#define __LINEAR_MODEL_TRAIN_NORMEQ_KERNEL_H__

#include "services/env_detect.h"
#include "data_management/data/numeric_table.h"
#include "src/data_management/service_numeric_table.h"

#include "src/algorithms/linear_model/linear_model_hyperparameter_impl.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace normal_equations
{
namespace training
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;

using namespace daal::algorithms::linear_model::internal;

/**
 * Abstract class that defines interface for the helper function that computes the regression coefficients.
 */
template <typename algorithmFPType, CpuType cpu>
class KernelHelperIface
{
public:
    /**
     * Computes regression coefficients by solving the system of linear equations
     * \param[in] p         Size of the system of linear equations
     * \param[in] a         Matrix of size P x P with semifinished left hand side of the system
     * \param[in,out] aCopy Auxiliary matrix of size P x P with a copy of the matrix a
     * \param[in] ny        Number of right hand sides of the system
     * \param[in,out] b     Matrix of size Ny x P.
     *                      On input, the right hand sides of the system of linear equations
     *                      On output, the regression coefficients
     * \param[in] interceptFlag Flag. If true, then it is required to compute an intercept term
     * \return Status of the computations
     */
    virtual Status computeBetasImpl(DAAL_INT p, const algorithmFPType * a, algorithmFPType * aCopy, DAAL_INT ny, algorithmFPType * b,
                                    bool inteceptFlag) const = 0;
};

/**
 * Implements the common part of the regression coefficients computation from partial result
 */
template <typename algorithmFPType, CpuType cpu>
class FinalizeKernel
{
    typedef WriteRows<algorithmFPType, cpu> WriteRowsType;
    typedef WriteOnlyRows<algorithmFPType, cpu> WriteOnlyRowsType;
    typedef ReadRows<algorithmFPType, cpu> ReadRowsType;

public:
    typedef linear_model::internal::Hyperparameter HyperparameterType;

    /**
     * Computes regression coefficients by solving the symmetric system of linear equations
     *      - X' - matrix of size N x P' that contains input data set of size N x P
     *             and optionally a column of 1's.
     *             Column of 1's is added when it is required to compute an intercept term
     *      - P' - number of columns in X'.
     *             P' = P + 1, when it is required to compute an intercept term;
     *             P' = P, otherwise
     * \param[in]  xtx      Input matrix \f$X'^T \times X'\f$ of size P' x P'
     * \param[in]  xty      Input matrix \f$X'^T \times Y\f$ of size Ny x P'
     * \param[out] xtxFinal Resulting matrix \f$X'^T \times X'\f$ of size P' x P'
     * \param[out] xtyFinal Resulting matrix \f$X'^T \times Y\f$ of size Ny x P'
     * \param[out] beta     Matrix with regression coefficients of size Ny x (P + 1)
     * \param[in]  interceptFlag    Flag. True if intercept term is not zero, false otherwise
     * \param[in]  helper   Object that implements the differences in the regression
     *                      coefficients computation
     * \return Status of the computations
     */
    static Status compute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                          bool interceptFlag, const KernelHelperIface<algorithmFPType, cpu> & helper);
    static Status compute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                          bool interceptFlag, const KernelHelperIface<algorithmFPType, cpu> & helper, const HyperparameterType * hyperparameter);

    static Status copyDataToTable(const algorithmFPType * data, size_t dataSizeInBytes, NumericTable & table);

    /**
     * Solves the symmetric system of linear equations
     * \param[in] p             Size of the system of linear equations
     * \param[in] a             Matrix of size P x P with the left hand side of the system
     * \param[in] ny            Number of right hand sides of the system
     * \param[in,out] b         Matrix of size Ny x P.
     *                          On input, the right hand sides of the system of linear equations
     *                          On output, the regression coefficients
     * \param[in] internalError Error code that have to be returned in case incorrect parameters
     *                          are passed into lapack routines
     * \return Status of the computations
     */
    static Status solveSystem(DAAL_INT p, algorithmFPType * a, DAAL_INT ny, algorithmFPType * b, const ErrorID & internalError);
};

/**
 * Thread local storage used on the partial results update stage
 */
template <typename algorithmFPType, CpuType cpu>
class ThreadingTask
{
    typedef ReadRows<algorithmFPType, cpu> ReadRowsType;

public:
    DAAL_NEW_DELETE();

    /**
     * Creates thread local storage of the requested size
     * \param[in] nBetasIntercept   Number of columms in the partial result
     * \param[in] nResponses        Number of responses
     * \return Pointer on the thread local storage object if the object was created successfully, NULL otherwise
     */
    static ThreadingTask<algorithmFPType, cpu> * create(size_t nBetasIntercept, size_t nResponses);
    virtual ~ThreadingTask();

    /**
     * Updates local partial result with the new block of data
     * \param[in] startRow  Index of the starting row of the block
     * \param[in] nRows     Number of rows in the block of data
     * \param[in] xTable    Input data set of size N x P
     * \param[in] yTable    Input array of responses of size N x Ny
     * \return Status of the computations
     */
    Status update(DAAL_INT startRow, DAAL_INT nRows, const NumericTable & xTable, const NumericTable & yTable);

    /**
     * Reduces thread local partial results into global partial result
     * \param[out] xtx Global partial result of size P' x P'
     * \param[out] xty Global partial result of size Ny x P'
     */
    void reduce(algorithmFPType * xtx, algorithmFPType * xty);

protected:
    /**
     * Construct thread local storage of the requested size
     * \param[in]  nBetasIntercept  Number of columns in the partial result
     * \param[in]  nResponses       Number of responses
     * \param[out] st               Status of the object construction
     */
    ThreadingTask(size_t nBetasIntercept, size_t nResponses, Status & st);
    algorithmFPType * _xtx;    /*!< Partial result of size P' x P' */
    algorithmFPType * _xty;    /*!< Partial result of size Ny x P' */
    ReadRowsType _xBlock;      /*!< Object that manages memory block of the input data set */
    ReadRowsType _yBlock;      /*!< Object that manages memory block of the input array of responses */
    DAAL_INT _nBetasIntercept; /*!< P' - number of columns in the partial result */
    DAAL_INT _nResponses;      /*!< Ny - number of responses */
};

/**
 * Implements the common part of the partial results update with new block of input data
 */
template <typename algorithmFPType, CpuType cpu>
class UpdateKernel
{
    typedef WriteRows<algorithmFPType, cpu> WriteRowsType;
    typedef ReadRows<algorithmFPType, cpu> ReadRowsType;
    typedef ThreadingTask<algorithmFPType, cpu> ThreadingTaskType;

public:
    typedef linear_model::internal::Hyperparameter HyperparameterType;

    /**
     * Updates normal equations model with the new block of data
     * \param[in]  x        Input data set of size N x P
     * \param[in]  y        Input responses of size N x Ny
     * \param[out] xtx      Matrix \f$X'^T \times X'\f$ of size P' x P'
     * \param[out] xty      Matrix \f$X'^T \times Y\f$ of size Ny x P'
     * \param[in]  initializeResult Flag. True if results initialization is required, false otherwise
     * \param[in]  interceptFlag    Flag.
     *                              - True if it is required to compute an intercept term and P' = P + 1
     *                              - False otherwise, P' = P
     * \return Status of the computations
     */
    static Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, bool initializeResult,
                          bool interceptFlag);
    static Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, bool initializeResult,
                          bool interceptFlag, const HyperparameterType * hyperparameter);
};

/**
 * Implements the common part of computations that merges together several partial results
 */
template <typename algorithmFPType, CpuType cpu>
class MergeKernel
{
    typedef WriteOnlyRows<algorithmFPType, cpu> WriteOnlyRowsType;
    typedef ReadRows<algorithmFPType, cpu> ReadRowsType;

public:
    typedef linear_model::internal::Hyperparameter HyperparameterType;

    /**
     * Merges an array of partial results into one partial result
     * \param[in] n          Number of partial results in the input array
     * \param[in] partialxtx Array of n numeric tables of size P x P
     * \param[in] partialxty Array of n numeric tables of size Ny x P
     * \param[out] xtx       Numeric table of size P x P
     * \param[out] xty       Numeric table of size Ny x P
     * \return Status of the computations
     */
    static Status compute(size_t n, NumericTable ** partialxtx, NumericTable ** partialxty, NumericTable & xtx, NumericTable & xty);
    static Status compute(size_t n, NumericTable ** partialxtx, NumericTable ** partialxty, NumericTable & xtx, NumericTable & xty,
                          const HyperparameterType * hyperparameter);

protected:
    /**
     * Adds input numeric table to the partial result
     * \param[in]  partialTable       Numeric table with partial sums
     * \param[out] result             Resulting array with full sums
     * \param[in]  threadingCondition Flag. If true, then the operation is performed in parallel
     * \return Status of the computations
     */
    static Status merge(const NumericTable & partialTable, algorithmFPType * result, bool threadingCondition);
};

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal
#endif
