/* file: dbscan_dense_batch.cpp */
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
!    C++ example of dense DBSCAN clustering in the batch processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DBSCAN_BATCH"></a>
 * \example dbscan_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/dbscan_dense.csv";

/* DBSCAN algorithm parameters */
const float epsilon = 0.04f;
const size_t minObservations = 45;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm object for the DBSCAN algorithm */
    dbscan::Batch<> algorithm(epsilon, minObservations);

    algorithm.input.set(dbscan::data, dataSource.getNumericTable());

    algorithm.compute();

    /* Print the clusterization results */
    printNumericTable(algorithm.getResult()->get(dbscan::nClusters), "Number of clusters:");
    printNumericTable(algorithm.getResult()->get(dbscan::assignments),
                      "Assignments of first 20 observations:",
                      20);

    return 0;
}
