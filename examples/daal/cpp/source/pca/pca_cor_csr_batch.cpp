/* file: pca_cor_csr_batch.cpp */
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
!    method in the batch processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_CSR_BATCH"></a>
 * \example pca_cor_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string dataFileName = "../data/batch/covcormoments_csr.csv";

typedef float algorithmFPType; /* Algorithm floating-point type */

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &dataFileName);

    /* Read data from a file and create a numeric table to store input data */
    CSRNumericTablePtr dataTable(createSparseTable<float>(dataFileName));

    /* Create an algorithm for principal component analysis using the correlation method */
    pca::Batch<> algorithm;

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    algorithm.parameter.covariance =
        services::SharedPtr<covariance::Batch<algorithmFPType, covariance::fastCSR> >(
            new covariance::Batch<algorithmFPType, covariance::fastCSR>());

    /* Set the algorithm input data */
    algorithm.input.set(pca::data, dataTable);

    algorithm.parameter.resultsToCompute = pca::mean | pca::variance | pca::eigenvalue;
    algorithm.parameter.isDeterministic = true;
    /* Compute results of the PCA algorithm */
    algorithm.compute();

    /* Print the results */
    pca::ResultPtr result = algorithm.getResult();
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");
    printNumericTable(result->get(pca::means), "Means:");
    printNumericTable(result->get(pca::variances), "Variances:");

    return 0;
}
