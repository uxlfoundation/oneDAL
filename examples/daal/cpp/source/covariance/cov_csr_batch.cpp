/* file: cov_csr_batch.cpp */
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
!    C++ example of variance-covariance matrix computation in the batch
!    processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_CSR_BATCH"></a>
 * \example cov_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters
   Input matrix is stored in the compressed sparse row format with one-based indexing
 */
const std::string datasetFileName = "../data/batch/covcormoments_csr.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create a numeric table to store input data */
    CSRNumericTablePtr dataTable(createSparseTable<float>(datasetFileName));

    /* Create an algorithm to compute variance-covariance matrix using the default method */
    covariance::Batch<float, covariance::fastCSR> algorithm;
    algorithm.input.set(covariance::data, dataTable);

    /* Compute a variance-covariance matrix */
    algorithm.compute();

    /* Get the computed variance-covariance matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::covariance),
                      "Covariance matrix (upper left square 10*10) :",
                      10,
                      10);
    printNumericTable(res->get(covariance::mean), "Mean vector:", 1, 10);

    return 0;
}
