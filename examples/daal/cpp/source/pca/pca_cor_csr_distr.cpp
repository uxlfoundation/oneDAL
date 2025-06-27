/* file: pca_cor_csr_distr.cpp */
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
!    method in the distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_CSR_DISTRIBUTED"></a>
 * \example pca_cor_csr_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileNames[] = { "../data/distributed/covcormoments_csr_1.csv",
                                         "../data/distributed/covcormoments_csr_2.csv",
                                         "../data/distributed/covcormoments_csr_3.csv",
                                         "../data/distributed/covcormoments_csr_4.csv" };

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    /* Create an algorithm for principal component analysis using the correlation method on the master node */
    pca::Distributed<step2Master> masterAlgorithm;

    for (size_t i = 0; i < nBlocks; i++) {
        CSRNumericTable* dataTable = createSparseTable<float>(datasetFileNames[i]);

        /* Create an algorithm to compute a variance-covariance matrix in the distributed processing mode using the default method */
        pca::Distributed<step1Local> localAlgorithm;

        /* Create an algorithm for principal component analysis using the correlation method on the local node */
        localAlgorithm.parameter.covariance = services::SharedPtr<
            covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR> >(
            new covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR>());

        /* Set input objects for the algorithm */
        localAlgorithm.input.set(pca::data, CSRNumericTablePtr(dataTable));

        /* Compute partial estimates on local nodes */
        localAlgorithm.compute();

        /* Set local partial results as input for the master-node algorithm */
        masterAlgorithm.input.add(pca::partialResults, localAlgorithm.getPartialResult());
    }

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    masterAlgorithm.parameter.covariance = services::SharedPtr<
        covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR> >(
        new covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR>());

    /* Merge and finalize PCA decomposition on the master node */
    masterAlgorithm.compute();

    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    pca::ResultPtr result = masterAlgorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");

    return 0;
}
