/* file: pca_cor_dense_distr.cpp */
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
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_DENSE_DISTRIBUTED"></a>
 * \example pca_cor_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileName = { "data/pca_normalized.csv" };

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Create an algorithm for principal component analysis using the correlation method on the master node */
    pca::Distributed<step2Master> masterAlgorithm;
    size_t totalRows = countRowsCSV(datasetFileName);
    size_t blockSize = (totalRows + nBlocks - 1) / nBlocks;

    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
    size_t remainingRows = totalRows;

    for (size_t i = 0; i < nBlocks; i++) {
        size_t rowsToRead = std::min(blockSize, remainingRows);
        size_t nLoaded = dataSource.loadDataBlock(rowsToRead);
        remainingRows -= nLoaded;

        /* Create an algorithm for principal component analysis using the correlation method on the local node */
        pca::Distributed<step1Local> localAlgorithm;

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
