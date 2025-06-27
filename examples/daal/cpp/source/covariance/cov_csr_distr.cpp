/* file: cov_csr_distr.cpp */
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
!    C++ example of variance-covariance matrix computation in the distributed
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_CSR_DISTRIBUTED">
 * \example cov_csr_distr.cpp
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

    printNumericTable(result->get(covariance::covariance),
                      "Covariance matrix (upper left square 10*10) :",
                      10,
                      10);
    printNumericTable(result->get(covariance::mean), "Mean vector:", 1, 10);

    return 0;
}

void computestep1Local(size_t block) {
    CSRNumericTable* dataTable = createSparseTable<float>(datasetFileNames[block]);

    /* Create an algorithm to compute a variance-covariance matrix in the distributed processing mode using the default method */
    covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(covariance::data, CSRNumericTablePtr(dataTable));

    /* Compute partial estimates on local nodes */
    algorithm.compute();

    /* Get the computed partial estimates */
    partialResult[block] = algorithm.getPartialResult();
}

void computeOnMasterNode() {
    /* Create an algorithm to compute a variance-covariance matrix in the distributed processing mode using the default method */
    covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR> algorithm;

    /* Set input objects for the algorithm */
    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(covariance::partialResults, partialResult[i]);
    }

    /* Compute a partial estimate on the master node from the partial estimates on local nodes */
    algorithm.compute();

    /* Finalize the result in the distributed processing mode */
    algorithm.finalizeCompute();

    /* Get the computed variance-covariance matrix */
    result = algorithm.getResult();
}
