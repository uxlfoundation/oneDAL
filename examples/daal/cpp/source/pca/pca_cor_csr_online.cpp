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
const std::string datasetFileNames[] = { "../data/online/covcormoments_csr_1.csv",
                                         "../data/online/covcormoments_csr_2.csv",
                                         "../data/online/covcormoments_csr_3.csv",
                                         "../data/online/covcormoments_csr_4.csv" };

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    /* Create an algorithm for principal component analysis using the correlation method */
    pca::Online<> algorithm;

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    algorithm.parameter.covariance =
        services::SharedPtr<covariance::Online<algorithmFPType, covariance::fastCSR> >(
            new covariance::Online<algorithmFPType, covariance::fastCSR>());

    for (size_t i = 0; i < nBlocks; i++) {
        /* Read data from a file and create a numeric table to store input data */
        CSRNumericTablePtr dataTable(createSparseTable<float>(datasetFileNames[i]));

        /* Set input objects for the algorithm */
        algorithm.input.set(pca::data, CSRNumericTablePtr(dataTable));

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
