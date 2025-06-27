/* file: cor_dense_distr.cpp */
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
!    C++ example of dense correlation matrix computation in the
!    distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CORRELATION_DENSE_DISTRIBUTED">
 * \example cor_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileNames[] = { "../data/distributed/covcormoments_dense_1.csv",
                                         "../data/distributed/covcormoments_dense_2.csv",
                                         "../data/distributed/covcormoments_dense_3.csv",
                                         "../data/distributed/covcormoments_dense_4.csv" };

covariance::PartialResultPtr partialResult[nBlocks];
covariance::ResultPtr result;

void computestep1Local(size_t i);
void computeOnMasterNode();

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    for (size_t i = 0; i < nBlocks; i++) {
        computestep1Local(i);
    }

    computeOnMasterNode();

    printNumericTable(result->get(covariance::correlation), "Correlation matrix:");
    printNumericTable(result->get(covariance::mean), "Mean vector:");

    return 0;
}

void computestep1Local(size_t block) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileNames[block],
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute a dense correlation matrix in the distributed processing mode using the default method */
    covariance::Distributed<step1Local> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(covariance::data, dataSource.getNumericTable());

    /* Compute partial estimates on local nodes */
    algorithm.compute();

    /* Get the computed partial estimates */
    partialResult[block] = algorithm.getPartialResult();
}

void computeOnMasterNode() {
    /* Create an algorithm to compute a dense correlation matrix in the distributed processing mode using the default method */
    covariance::Distributed<step2Master> algorithm;

    /* Set input objects for the algorithm */
    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(covariance::partialResults, partialResult[i]);
    }

    /* Set the parameter to choose the type of the output matrix */
    algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

    /* Compute a partial estimate on the master node from the partial estimates on local nodes */
    algorithm.compute();

    /* Finalize the result in the distributed processing mode */
    algorithm.finalizeCompute();

    /* Get the computed dense correlation matrix */
    result = algorithm.getResult();
}
