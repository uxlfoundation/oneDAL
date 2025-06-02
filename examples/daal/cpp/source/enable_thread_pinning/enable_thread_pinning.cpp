/* file: enable_thread_pinning.cpp */
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
!    C++ example of thread pinning usage
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ENABLE_THREAD_PINNING"></a>
 * \example enable_thread_pinning.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
std::string datasetFileName = "../data/batch/kmeans_dense.csv";

/* K-Means algorithm parameters */
const size_t nClusters = 20;
const size_t nIterations = 5;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Get initial clusters for the K-Means algorithm */
    kmeans::init::Batch<float, kmeans::init::randomDense> init(nClusters);

    init.input.set(kmeans::init::data, dataSource.getNumericTable());

    /* Enables thread pinning for next algorithm runs */
    services::Environment::getInstance()->enableThreadPinning(true);

    init.compute();

    /* Disables thread pinning for next algorithm runs */
    services::Environment::getInstance()->enableThreadPinning(false);

    NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm */
    kmeans::Batch<> algorithm(nClusters, nIterations);

    algorithm.input.set(kmeans::data, dataSource.getNumericTable());
    algorithm.input.set(kmeans::inputCentroids, centroids);

    /* Run computations */
    algorithm.compute();

    /* Print the clusterization results */
    printNumericTable(algorithm.getResult()->get(kmeans::assignments),
                      "First 10 cluster assignments:",
                      10);
    printNumericTable(algorithm.getResult()->get(kmeans::centroids),
                      "First 10 dimensions of centroids:",
                      20,
                      10);
    printNumericTable(algorithm.getResult()->get(kmeans::objectiveFunction),
                      "Objective function value:");

    return 0;
}
