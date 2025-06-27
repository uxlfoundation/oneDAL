/* file: kmeans_init_dense_batch.cpp */
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
!    C++ example of dense K-Means clustering with different initialization methods
!    in the batch processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_INIT_DENSE_BATCH"></a>
 * \example kmeans_init_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/kmeans_init_dense.csv";

/* K-Means algorithm parameters */
const size_t nMaxIterations = 1000;
const double cAccuracyThreshold = 0.01;
const size_t nClusters = 20;

template <typename Type>
Type getSingleValue(const NumericTablePtr& pTbl) {
    BlockDescriptor<Type> block;
    pTbl->getBlockOfRows(0, 1, readOnly, block);
    Type value = block.getBlockPtr()[0];
    pTbl->releaseBlockOfRows(block);
    return value;
}

template <kmeans::init::Method method>
static void runKmeans(const NumericTablePtr& inputData,
                      size_t nClusters,
                      const char* methodName,
                      double oversamplingFactor = -1.0) {
    /* Get initial clusters for the K-Means algorithm */
    kmeans::init::Batch<float, method> init(nClusters);
    init.input.set(kmeans::init::data, inputData);
    init.parameter.nTrials = 1;
    if (oversamplingFactor > 0)
        init.parameter.oversamplingFactor = oversamplingFactor;
    std::cout << "K-means init parameters: method = " << methodName;
    if (method == kmeans::init::parallelPlusDense)
        std::cout << ", oversamplingFactor = " << init.parameter.oversamplingFactor
                  << ", nRounds = " << init.parameter.nRounds;
    std::cout << std::endl;

    init.compute();
    NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm */
    kmeans::Batch<> algorithm(nClusters, nMaxIterations);

    algorithm.input.set(kmeans::data, inputData);
    algorithm.input.set(kmeans::inputCentroids, centroids);
    algorithm.parameter().accuracyThreshold = cAccuracyThreshold;
    std::cout << "K-means algorithm parameters: maxIterations = "
              << algorithm.parameter().maxIterations
              << ", accuracyThreshold = " << algorithm.parameter().accuracyThreshold << std::endl;
    algorithm.compute();

    /* Print the results */
    const float goalFunc =
        getSingleValue<float>(algorithm.getResult()->get(kmeans::objectiveFunction));
    const int nIterations = getSingleValue<int>(algorithm.getResult()->get(kmeans::nIterations));
    std::cout << "K-means algorithm results: Objective function value = " << goalFunc * 1e-6
              << "*1E+6, number of iterations = " << nIterations << std::endl
              << std::endl;
}

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();
    NumericTablePtr inputData = dataSource.getNumericTable();

    runKmeans<kmeans::init::deterministicDense>(inputData, nClusters, "deterministicDense");
    runKmeans<kmeans::init::randomDense>(inputData, nClusters, "randomDense");
    runKmeans<kmeans::init::plusPlusDense>(inputData, nClusters, "plusPlusDense");
    runKmeans<kmeans::init::parallelPlusDense>(inputData, nClusters, "parallelPlusDense", 0.5);
    runKmeans<kmeans::init::parallelPlusDense>(inputData, nClusters, "parallelPlusDense", 2.0);
    return 0;
}
