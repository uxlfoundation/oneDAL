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

template <typename algorithmFPType, CpuType cpu>
class LinearModelReducer : public daal::Reducer
{
public:
    /// Construct and initialize the thread-local partial results
    ///
    /// @param[in] xTable        Input data table that stores matrix X with feature vectors
    /// @param[in] yTable        Input data table that stores matrix Y with target values
    /// @param[in] nBetasIntercept   Number of trainable parameters
    /// @param[in] numRowsInBlock   Number of rows in the block of the input data table - a mininal number of rows to be processed by a thread
    /// @param[in] numBlocks        Number of blocks of rows in the input data table
    LinearModelReducer(const NumericTable & xTable, const NumericTable & yTable, DAAL_INT nBetasIntercept, DAAL_INT numRowsInBlock, size_t numBlocks);

    /// New and delete operators are overloaded to use scalable memory allocator that doesn't block threads
    /// if memory allocations are executed concurrently.
    void * operator new(size_t size);
    void operator delete(void * p);

    /// Get pointer to the array of partial X^tY matrix.
    inline algorithmFPType * xty();
    /// Get pointer to the partial X^tX matrix.
    inline algorithmFPType * xtx();

    /// Get constant pointer to the array of partial X^tY matrix.
    inline const algorithmFPType * xty() const;
    /// Get constant pointer to the partial X^tX matrix.
    inline const algorithmFPType * xtx() const;

    /// Constructs a thread-local partial result and initializes it with zeros.
    /// Must be able to run concurrently with `update` and `join` methods.
    ///
    /// @return Pointer to the partial result of the linear regression algorithm.
    virtual ReducerUniquePtr create() const override;

    /// Updates partial X^tX and X^tY matrices
    /// from the blocks of input data table in the sub-interval [begin, end).
    ///
    /// @param begin Index of the starting block of the input data table.
    /// @param end   Index of the block after the last one in the sub-range.
    virtual void update(size_t begin, size_t end) override;

    /// Merge the partial result with the data from another thread.
    ///
    /// @param otherReducer Pointer to the other thread's partial result.
    virtual void join(Reducer * otherReducer) override;

private:
    /// Reference to the input data table that stores matrix X with feature vectors.
    const NumericTable & _xTable;
    /// Reference to the input data table that stores matrix Y with target values.
    const NumericTable & _yTable;
    /// Number of features in the input data table.
    DAAL_INT _nFeatures;
    /// Number of targets to predict.
    DAAL_INT _nResponses;
    /// Number of trainable parameters (including intercept).
    DAAL_INT _nBetasIntercept;
    /// Number of rows in the block of the input data table - a mininal number of rows to be processed by a thread.
    DAAL_INT _numRowsInBlock;
    /// Number of blocks of rows in the input data table.
    size_t _numBlocks;
    /// Thread-local array of X^tX partial sums of size `_nBetasIntercept` * `_nBetasIntercept`.
    TArrayScalableCalloc<algorithmFPType, cpu> _xtx;
    /// Thread-local array of X^tY partial sums of size `_nBetasIntercept` * `_nLabels`.
    TArrayScalableCalloc<algorithmFPType, cpu> _xty;
};

/**
 * Implements the common part of the partial results update with new block of input data
 */
template <typename algorithmFPType, CpuType cpu>
class UpdateKernel
{
    typedef WriteRows<algorithmFPType, cpu> WriteRowsType;
    typedef ReadRows<algorithmFPType, cpu> ReadRowsType;
    // typedef ThreadingTask<algorithmFPType, cpu> ThreadingTaskType;

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
     *                              - False otherwis, P' = P
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
     * \param[in] n          Number of partial resuts in the input array
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
