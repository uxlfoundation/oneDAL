/* file: spectral_embedding_default_dense_impl.i */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
//  Implementation of cosine distance.
//--
*/

#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/cosdistance/cosdistance_kernel.h"
#include "src/externals/service_lapack.h"
#include <iostream>

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace spectral_embedding
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
services::Status computeEigenvectorsInplace(size_t nFeatures, algorithmFPType * eigenvectors, algorithmFPType * eigenvalues)
{
    char jobz = 'V';
    char uplo = 'U';

    DAAL_INT lwork  = 2 * nFeatures * nFeatures + 6 * nFeatures + 1;
    DAAL_INT liwork = 5 * nFeatures + 3;
    DAAL_INT info;

    TArray<algorithmFPType, cpu> work(lwork);
    TArray<DAAL_INT, cpu> iwork(liwork);
    DAAL_CHECK_MALLOC(work.get() && iwork.get());

    LapackInst<algorithmFPType, cpu>::xsyevd(&jobz, &uplo, (DAAL_INT *)(&nFeatures), eigenvectors, (DAAL_INT *)(&nFeatures), eigenvalues, work.get(),
                                             &lwork, iwork.get(), &liwork, &info);
    if (info != 0) return services::Status(services::ErrorPCAFailedToComputeCorrelationEigenvalues); // CHANGE ERROR STATUS
    return services::Status();
}

/**
 *  \brief Kernel for Spectral Embedding calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status SpectralEmbeddingKernel<algorithmFPType, method, cpu>::compute(const NumericTable * xTable, NumericTable * embeddingTable,
                                                                                NumericTable * eigenTable, const KernelParameter & par)
{
    services::Status status;
    // std::cout << "inside DAAL kernel" << std::endl;
    // std::cout << "Params: " << par.numberOfEmbeddings << " " << par.numberOfNeighbors << std::endl;
    size_t k       = par.numberOfEmbeddings;
    size_t filtNum = par.numberOfNeighbors + 1;
    size_t n       = xTable->getNumberOfRows(); /* Number of input feature vectors   */

    SharedPtr<HomogenNumericTable<algorithmFPType> > tmpMatrixPtr =
        HomogenNumericTable<algorithmFPType>::create(n, n, NumericTable::doAllocate, &status);

    DAAL_CHECK_STATUS_VAR(status);
    NumericTable * covOutput   = tmpMatrixPtr.get();
    NumericTable * a0          = const_cast<NumericTable *>(xTable);
    NumericTable * eigenvalues = const_cast<NumericTable *>(eigenTable);

    // Compute cosine distances matrix
    {
        auto cosDistanceKernel = cosine_distance::internal::DistanceKernel<algorithmFPType, cosine_distance::Method::defaultDense, cpu>();
        DAAL_CHECK_STATUS(status, cosDistanceKernel.compute(0, &a0, 0, &covOutput, nullptr));
    }

    WriteRows<algorithmFPType, cpu> xMatrix(covOutput, 0, n);
    DAAL_CHECK_BLOCK_STATUS(xMatrix);
    algorithmFPType * x = xMatrix.get();

    size_t lcnt, rcnt, cnt;
    algorithmFPType L, R, M;
    // Use binary search to find such d that the number of verticies having distance <= d is filtNum
    const size_t binarySearchIterNum = 20;
    // TODO: add parallel_for
    for (size_t i = 0; i < n; ++i)
    {
        L    = 0; // min possible cos distance
        R    = 2; // max possible cos distance
        lcnt = 0; // number of elements with cos distance <= L
        rcnt = n; // number of elements with cos distance <= R
        for (size_t ij = 0; ij < binarySearchIterNum; ++ij)
        {
            M   = (L + R) / 2;
            cnt = 0;
            // Calculate the number of elements in the row with value <= M
            for (size_t j = 0; j < n; ++j)
            {
                if (x[i * n + j] <= M)
                {
                    cnt++;
                }
            }
            if (cnt < filtNum)
            {
                L    = M;
                lcnt = cnt;
            }
            else
            {
                R    = M;
                rcnt = cnt;
            }
            // distance threshold is found
            if (rcnt == filtNum)
            {
                break;
            }
        }
        // create edges for the closest neighbors
        for (size_t j = 0; j < n; ++j)
        {
            if (x[i * n + j] <= R)
            {
                x[i * n + j] = 1.0;
            }
            else
            {
                x[i * n + j] = 0.0;
            }
        }
        // fill the diagonal of matrix with zeros
        x[i * n + i] = 0;
    }

    // Create Laplassian matrix
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            algorithmFPType val = (x[i * n + j] + x[j * n + i]) / 2;
            x[i * n + j]        = -val;
            x[j * n + i]        = -val;
            x[i * n + i] += val;
            x[j * n + j] += val;
        }
    }

    // std::cout << "Laplacian matrix" << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << x[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "------" << std::endl;

    // Find the eigen vectors and eigne values of the matix
    //TArray<algorithmFPType, cpu> eigenvalues(n);
    //DAAL_CHECK_MALLOC(eigenvalues.get());
    WriteRows<algorithmFPType, cpu> eigenValuesBlock(eigenvalues, 0, n);
    DAAL_CHECK_BLOCK_STATUS(eigenValuesBlock);
    algorithmFPType * eigenValuesPtr = eigenValuesBlock.get();

    status |= computeEigenvectorsInplace<algorithmFPType, cpu>(n, x, eigenValuesPtr);
    DAAL_CHECK_STATUS_VAR(status);

    // std::cout << "Eigen vectors: " << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << x[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Fill the output matrix with eigen vectors corresponding to the smallest eigen values
    WriteOnlyRows<algorithmFPType, cpu> embedMatrix(embeddingTable, 0, n);
    DAAL_CHECK_BLOCK_STATUS(embedMatrix);
    algorithmFPType * embed = embedMatrix.get();

    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            embed[j * k + i] = x[i * n + j];
        }
    }

    return status;
}

} // namespace internal

} // namespace spectral_embedding

} // namespace algorithms

} // namespace daal
