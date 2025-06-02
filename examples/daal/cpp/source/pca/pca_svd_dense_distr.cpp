/* file: pca_svd_dense_distr.cpp */
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
!    C++ example of principal component analysis (PCA) using the singular value
!    decomposition (SVD) method in the distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_SVD_DENSE_DISTRIBUTED"></a>
 * \example pca_svd_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;
const size_t nVectorsInBlock = 250;
size_t nFeatures;

const std::string dataFileNames[] = { "../data/distributed/pca_normalized_1.csv",
                                      "../data/distributed/pca_normalized_2.csv",
                                      "../data/distributed/pca_normalized_3.csv",
                                      "../data/distributed/pca_normalized_4.csv" };

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &dataFileNames[0],
                   &dataFileNames[1],
                   &dataFileNames[2],
                   &dataFileNames[3]);

    /* Create an algorithm for principal component analysis using the SVD method on the master node */
    pca::Distributed<step2Master, algorithmFPType, pca::svdDense> masterAlgorithm;

    for (size_t i = 0; i < nBlocks; i++) {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(dataFileNames[i],
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

        /* Retrieve the input data */
        dataSource.loadDataBlock(nVectorsInBlock);

        /* Create an algorithm for principal component analysis using the SVD method on the local node */
        pca::Distributed<step1Local, algorithmFPType, pca::svdDense> localAlgorithm;

        /* Set the input data to the algorithm */
        localAlgorithm.input.set(pca::data, dataSource.getNumericTable());

        /* Compute PCA decomposition */
        localAlgorithm.compute();

        /* Set local partial results as input for the master-node algorithm */
        masterAlgorithm.input.add(pca::partialResults, localAlgorithm.getPartialResult());
    }

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
