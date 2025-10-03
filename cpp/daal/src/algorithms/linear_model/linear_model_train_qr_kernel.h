/* file: linear_model_train_qr_kernel.h */
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

#ifndef __LINEAR_MODEL_TRAIN_QR_KERNEL_H__
#define __LINEAR_MODEL_TRAIN_QR_KERNEL_H__

#include "services/env_detect.h"
#include "data_management/data/numeric_table.h"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace qr
{
namespace training
{
namespace internal
{
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services::internal;

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
     * \param[in] nRows             Number of rows in the partial result
     * \param[in] nResponses        Number of responses
     * \return Pointer on the thread local storage object if the object was created successfully, NULL otherwise
     */
    static ThreadingTask<algorithmFPType, cpu> * create(DAAL_INT nBetasIntercept, DAAL_INT nRows, DAAL_INT nResponses);

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
     * \param[out] r    Global partial result of size P' x P'
     * \param[out] qty  Global partial result of size Ny x P'
     */
    Status reduce(algorithmFPType * r, algorithmFPType * qty);

    /**
     * Reduces thread local and global partial results into global partial result
     * \param[out] r    Global partial result of size P' x P'
     * \param[out] qty  Global partial result of size Ny x P'
     */
    Status reduceInPlace(algorithmFPType * r, algorithmFPType * qty);

protected:
    DAAL_INT _lwork;                                     /*!< Size of work array for lapack routines */
    TArrayScalable<algorithmFPType, cpu> work;           /*!< Work array for lapack routines */
    TArrayScalable<algorithmFPType, cpu> tau;            /*!< Array of scalar factors of the elementary
                                                             reflectors for the matrix Q */
    TArrayScalable<algorithmFPType, cpu> qrBuffer;       /*!< Auxiliary array of size Npart x P
                                                             that stores the copy of input matrix part */
    TArrayScalable<algorithmFPType, cpu> qtyBuffer;      /*!< Auxiliary array of size Ny x P
                                                             that stores the copy of input matrix part*/
    TArrayScalableCalloc<algorithmFPType, cpu> qrR;      /*!< Array of size P' x P' that stores matrix R
                                                             accumulated on this thread */
    TArrayScalableCalloc<algorithmFPType, cpu> qrQTY;    /*!< Array of size Ny x P' that stores matrix Q^T * Y
                                                             accumulated on this thread */
    TArrayScalableCalloc<algorithmFPType, cpu> qrRNew;   /*!< Array of size P' x P' that stores matrix R
                                                             obtained from the current block of data */
    TArrayScalableCalloc<algorithmFPType, cpu> qrQTYNew; /*!< Array of size Ny x P' that stores matrix Q^T * Y
                                                             obtained from the current block of data */
    TArrayScalable<algorithmFPType, cpu> qrRMerge;       /*!< Array of size 2*P' x P' that stores matrices
                                                             qrR and qrRNew merged by rows */
    TArrayScalable<algorithmFPType, cpu> qrQTYMerge;     /*!< Array of size 2*Ny x P' that stores matrices
                                                             qrQTY and qrQTYNew merged by rows */
    ReadRowsType _xBlock;                                /*!< Object that manages memory block of the input data set */
    ReadRowsType _yBlock;                                /*!< Object that manages memory block of the input array of responses */

    DAAL_INT _nBetasIntercept; /*!< Number of rows and columns in the matrix R */
    DAAL_INT _nRows;           /*!< Npart, number of rows in the input data set part */
    DAAL_INT _nResponses;      /*!< Number of responses */

    /**
     * Construct thread local storage of the requested size
     * \param[in]  nBetasIntercept  Number of columms in the partial result
     * \param[in]  nRows            Number of rows in the partial result
     * \param[in]  nResponses       Number of responses
     * \param[out] st               Status of the object construction
     */
    ThreadingTask(DAAL_INT nBetasIntercept, DAAL_INT nRows, DAAL_INT nResponses, Status & st);

    /**
     * Allocates work buffer for lapack routines
     * \return Status of memory allocation
     */
    Status allocateWorkBuffer();

    /**
     * Copy block of rows from input data set into thread local storage
     * \param[in] startRow  Index of the starting row of the block
     * \param[in] nRows     Number of rows in the block of data
     * \param[in] xTable    Input data set of size N x P
     * \param[in] yTable    Input array of responses of size N x Ny
     * \return Status of the computations
     */
    Status copyDataToBuffer(DAAL_INT startRow, DAAL_INT nRows, const NumericTable & xTable, const NumericTable & yTable);
};

/**
 * Implements the common part of the update and merge stages of the QR method
 * for linear regression training
 */
template <typename algorithmFPType, CpuType cpu>
class CommonKernel
{
public:
    /**
     * Computes work buffer size for lapack GERQF and ORMRQ routines
     * \param[in]  nRows      Number of rows in the GERQF input matrix
     * \param[in]  nCols      Number of columns in the GERQF input matrix
     * \param[in]  nResponses Number of rows in the ORMRQ input matrix
     * \param[out] lwork      Size of the work buffer
     * \return Status of the computations
     */
    static Status computeWorkSize(DAAL_INT nRows, DAAL_INT nCols, DAAL_INT nResponses, DAAL_INT & lwork);

    /**
     * Computes QR decomposition of input matrix
     * \param[in]  p     Number of columns in the input matrix
     * \param[in]  n     Number of rows in the input matrix
     * \param[in]  x     Input matrix of size n x p
     * \param[in]  ny    Number of responses
     * \param[in]  y     Input array of responses of size ny x p
     * \param[out] r     Matrix R of size p x p
     * \param[out] qty   Matrix Q^T * Y of size ny x p
     * \param[out] tau   Array of scalar factors of the elementary
     *                   reflectors for the matrix Q
     * \param[in]  work  Work buffer for lapack routines
     * \param[in]  lwork Size of the work buffer
     * \return Status of the computations
     */
    static Status computeQRForBlock(DAAL_INT p, DAAL_INT n, const algorithmFPType * x, DAAL_INT ny, const algorithmFPType * y, algorithmFPType * r,
                                    algorithmFPType * qty, algorithmFPType * tau, algorithmFPType * work, DAAL_INT lwork);

    /**
     * Merges two results of QR decomposition together
     * \param[in]  p     Size of the matrix R
     * \param[in]  ny    Number of rows in the matrix Q^T * Y
     * \param[in]  r1    Matrix R1 of size p x p, result of the 1st QR decomposition
     * \param[in]  qty1  Matrix Q1^T * Y1 of size ny x p, result of the 1st QR decomposition
     * \param[in]  r2    Matrix R2 of size p x p, result of the 2nd QR decomposition
     * \param[in]  qty2  Matrix Q2^T * Y2 of size ny x p, result of the 2nd QR decomposition
     * \param[out] r12   Matrix of size 2*p x p that holds matrices R1 and R2 merged by rows
     * \param[out] qty12 Matrix of size 2*ny x p that holds matrices Q1^T * Y1 and Q2^T * Y2 merged by rows
     * \param[out] r     Resulting matrix R of size p x p
     * \param[out] qty   Resulting Q^T * Y of size ny x p
     * \param[out] tau   Array of scalar factors of the elementary
     *                   reflectors for the matrix Q
     * \param[in]  work  Work buffer for lapack routines
     * \param[in]  lwork Size of the work buffer
     */
    static Status merge(DAAL_INT p, DAAL_INT ny, const algorithmFPType * r1, const algorithmFPType * qty1, const algorithmFPType * r2,
                        const algorithmFPType * qty2, algorithmFPType * r12, algorithmFPType * qty12, algorithmFPType * r, algorithmFPType * qty,
                        algorithmFPType * tau, algorithmFPType * work, DAAL_INT lwork);
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
    /**
     * Updates QR model with the new block of data
     * \param[in]  x        Input data set of size N x P
     * \param[in]  y        Input responses of size N x Ny
     * \param[out] r        Matrix R of size P' x P'
     * \param[out] qty      Matrix \f$Q^T \times Y\f$ of size Ny x P'
     * \param[in]  initializeResult Flag. True if results initialization is required, false otherwise
     * \param[in]  interceptFlag    Flag.
     *                              - True if it is required to compute an intercept term and P' = P + 1
     *                              - False otherwise, P' = P
     * \return Status of the computations
     */
    static Status compute(const NumericTable & x, const NumericTable & y, NumericTable & r, NumericTable & qty, bool initializeResult,
                          bool interceptFlag);
};

/**
 * Implements the common part of computations that merges together several partial results
 */
template <typename algorithmFPType, CpuType cpu>
class MergeKernel
{
    typedef WriteRows<algorithmFPType, cpu> WriteRowsType;
    typedef ReadRows<algorithmFPType, cpu> ReadRowsType;

public:
    /**
     * Merges an array of partial results into one partial result
     * \param[in] n          Number of partial results in the input array
     * \param[in] partialr   Array of n numeric tables R of size P x P
     * \param[in] partialqty Array of n numeric tables \f$Q^T \times Y\f$ of size Ny x P
     * \param[out] r         Numeric table R of size P x P
     * \param[out] xty       Numeric table \f$Q^T \times Y\f$ of size Ny x P
     * \return Status of the computations
     */
    static Status compute(size_t n, NumericTable ** partialr, NumericTable ** partialqty, NumericTable & r, NumericTable & qty);
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
    /**
     * Computes regression coefficients by solving triangular system of linear equations
     *      - X' - matrix of size N x P' that contains input data set of size N x P
     *             and optionally a column of 1's.
     *             Column of 1's is added when it is required to compute an intercept term
     *      - P' - number of columns in X'.
     *             P' = P + 1, when it is required to compute an intercept term;
     *             P' = P, otherwise
     * \param[in]  r        Input matrix R of size P' x P'
     * \param[in]  qty      Input matrix \f$Q^T \times Y\f$ of size Ny x P'
     * \param[out] rFinal   Resulting matrix R of size P' x P'
     * \param[out] qtyFinal Resulting matrix \f$Q^T \times Y\f$ of size Ny x P'
     * \param[out] beta     Matrix with regression coefficients of size Ny x (P + 1)
     * \param[in]  interceptFlag    Flag. True if intercept term is not zero, false otherwise
     * \return Status of the computations
     */
    static Status compute(const NumericTable & r, const NumericTable & qty, NumericTable & rFinal, NumericTable & qtyFinal, NumericTable & beta,
                          bool interceptFlag);
};

} // namespace internal
} // namespace training
} // namespace qr
} // namespace linear_model
} // namespace algorithms
} // namespace daal
#endif
