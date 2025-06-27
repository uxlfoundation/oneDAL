/* file: cov_dense_online.cpp */
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
!    C++ example of dense variance-covariance matrix computation in the online
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_DENSE_ONLINE"></a>
 * \example cov_dense_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/covcormoments_dense.csv";
const size_t nObservations = 50;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create an algorithm to compute a dense variance-covariance matrix in the online processing mode using the default method */
    covariance::Online<> algorithm;
    while (dataSource.loadDataBlock(nObservations) == nObservations) {
        /* Set input objects for the algorithm */
        algorithm.input.set(covariance::data, dataSource.getNumericTable());

        /* Compute partial estimates */
        algorithm.compute();
    }

    /* Finalize the result in the online processing mode */
    algorithm.finalizeCompute();

    /* Get the computed dense variance-covariance matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::covariance), "Covariance matrix:");
    printNumericTable(res->get(covariance::mean), "Mean vector:");

    return 0;
}
