/* file: set_number_of_threads.cpp */
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
!    C++ example of setting the maximum number of threads
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SET_NUMBER_OF_THREADS"></a>
 * \example set_number_of_threads.cpp
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
const size_t nThreads = 2;
size_t nThreadsInit;
size_t nThreadsNew;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Get the number of threads that is used by the library by default */
    nThreadsInit = services::Environment::getInstance()->getNumberOfThreads();

    /* Set the maximum number of threads to be used by the library */
    services::Environment::getInstance()->setNumberOfThreads(nThreads);

    /* Get the number of threads that is used by the library after changing */
    nThreadsNew = services::Environment::getInstance()->getNumberOfThreads();

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Get initial clusters for the K-Means algorithm */
    kmeans::init::Batch<float, kmeans::init::randomDense> init(nClusters);

    init.input.set(kmeans::init::data, dataSource.getNumericTable());
    init.compute();

    NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm */
    kmeans::Batch<> algorithm(nClusters, nIterations);

    algorithm.input.set(kmeans::data, dataSource.getNumericTable());
    algorithm.input.set(kmeans::inputCentroids, centroids);

    /* Run computations */
    algorithm.compute();

    std::cout << "Initial number of threads:        " << nThreadsInit << std::endl;
    std::cout << "Number of threads to set:         " << nThreads << std::endl;
    std::cout << "Number of threads after setting:  " << nThreadsNew << std::endl;

    return 0;
}
