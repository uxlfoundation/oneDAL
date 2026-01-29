/* file: cov_dense_distr.cpp */
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
!    C++ example of dense variance-covariance matrix computation in the
!    distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_DENSE_DISTRIBUTED">
 * \example cov_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileName = { "data/covcormoments_dense.csv" };

covariance::PartialResultPtr partialResult[nBlocks];
covariance::ResultPtr result;

void computestep1Local(size_t block, const NumericTablePtr &data);
void computeOnMasterNode();

int main(int argc, char *argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    size_t totalRows = countRowsCSV(datasetFileName);
    size_t blockSize = (totalRows + nBlocks - 1) / nBlocks;

    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
    size_t remainingRows = totalRows;

    for (size_t block = 0; block < nBlocks && remainingRows > 0; block++) {
        size_t rowsToRead = std::min(blockSize, remainingRows);
        size_t nLoaded = dataSource.loadDataBlock(rowsToRead);
        remainingRows -= nLoaded;
        NumericTablePtr blockTable = dataSource.getNumericTable();
        computestep1Local(block, blockTable);
    }

    computeOnMasterNode();

    printNumericTable(result->get(covariance::covariance), "Covariance matrix:");
    printNumericTable(result->get(covariance::mean), "Mean vector:");

    return 0;
}

void computestep1Local(size_t block, const NumericTablePtr &data) {
    /* Create an algorithm to compute a dense correlation matrix in the distributed processing mode using the default method */
    covariance::Distributed<step1Local> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(covariance::data, data);

    /* Compute partial estimates on local nodes */
    algorithm.compute();

    /* Get the computed partial estimates */
    partialResult[block] = algorithm.getPartialResult();
}

void computeOnMasterNode() {
    /* Create an algorithm to compute a dense variance-covariance matrix in the distributed processing mode using the default method */
    covariance::Distributed<step2Master> algorithm;

    /* Set input objects for the algorithm */
    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(covariance::partialResults, partialResult[i]);
    }

    /* Compute a partial estimate on the master node from the partial estimates on local nodes */
    algorithm.compute();

    /* Finalize the result in the distributed processing mode */
    algorithm.finalizeCompute();

    /* Get the computed dense variance-covariance matrix */
    result = algorithm.getResult();
}
