/* file: qr_dense_online.cpp */
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
!    C++ example of computing QR decomposition in the online processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-QR_ONLINE"></a>
 * \example qr_dense_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/online/qr.csv";
const size_t nRowsInBlock = 4000;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create an algorithm to compute QR decomposition in the online processing mode */
    qr::Online<> algorithm;

    while (dataSource.loadDataBlock(nRowsInBlock) == nRowsInBlock) {
        algorithm.input.set(qr::data, dataSource.getNumericTable());

        /* Compute QR decomposition */
        algorithm.compute();
    }

    /* Finalize computations and retrieve the results */
    algorithm.finalizeCompute();

    qr::ResultPtr res = algorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(qr::matrixQ), "Orthogonal matrix Q:", 10);
    printNumericTable(res->get(qr::matrixR), "Triangular matrix R:");

    return 0;
}
