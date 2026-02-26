/* file: pca_cor_csr_online.cpp */
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
!  Content:
!    C++ example of principal component analysis (PCA) using the correlation
!    method in the online processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_CSR_ONLINE"></a>
 * \example pca_cor_csr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;
const std::string datasetFileName = "data/covcormoments_csr.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    CSRNumericTablePtr fullData(createSparseTable<algorithmFPType>(datasetFileName));

    const size_t totalRows = fullData->getNumberOfRows();
    const size_t rowsPerBlock = (totalRows + nBlocks - 1) / nBlocks;

    /* Create an algorithm for principal component analysis using the correlation method */
    pca::Online<> algorithm;

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    algorithm.parameter.covariance =
        services::SharedPtr<covariance::Online<algorithmFPType, covariance::fastCSR> >(
            new covariance::Online<algorithmFPType, covariance::fastCSR>());

    for (size_t block = 0; block < nBlocks; ++block) {
        const size_t rowStart = block * rowsPerBlock;
        const size_t rowEnd = std::min(rowStart + rowsPerBlock, totalRows);

        /* Split CSR exactly like in distributed */
        CSRNumericTablePtr localTable = splitCSRBlock<algorithmFPType>(fullData, rowStart, rowEnd);

        /* Set input objects for the algorithm */
        algorithm.input.set(pca::data, localTable);

        /* Update PCA decomposition */
        algorithm.compute();
    }

    /* Finalize computations */
    algorithm.finalizeCompute();

    /* Print the results */
    pca::ResultPtr result = algorithm.getResult();
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");

    return 0;
}
