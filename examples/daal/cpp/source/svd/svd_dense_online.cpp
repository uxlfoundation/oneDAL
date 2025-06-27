/* file: svd_dense_online.cpp */
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
!    C++ example of singular value decomposition (SVD) in the online processing
!    mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVD_ONLINE"></a>
 * \example svd_dense_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/online/svd.csv";
const size_t nRowsInBlock = 4000;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create an algorithm to compute SVD in the online processing mode */
    svd::Online<> algorithm;

    while (dataSource.loadDataBlock(nRowsInBlock) == nRowsInBlock) {
        algorithm.input.set(svd::data, dataSource.getNumericTable());

        /* Compute SVD */
        algorithm.compute();
    }

    /* Finalize computations and retrieve the results */
    algorithm.finalizeCompute();

    svd::ResultPtr res = algorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(svd::singularValues), "Singular values:");
    printNumericTable(res->get(svd::rightSingularMatrix), "Right orthogonal matrix V:");
    printNumericTable(res->get(svd::leftSingularMatrix), "Left orthogonal matrix U:", 10);

    return 0;
}
