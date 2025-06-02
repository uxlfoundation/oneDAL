/* file: kmeans_csr_batch_assign.cpp */
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
!    C++ example of sparse K-Means clustering in the batch processing mode
!    for calculation assignments without centroids update
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_CSR_BATCH_ASSIGN"></a>
 * \example kmeans_csr_batch_assign.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
std::string datasetFileName = "../data/batch/kmeans_csr.csv";

/* K-Means algorithm parameters */
const size_t nClusters = 20;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Retrieve the data from the input file */
    CSRNumericTablePtr dataTable(createSparseTable<float>(datasetFileName));

    /* Get initial clusters for the K-Means algorithm */
    kmeans::init::Batch<algorithmFPType, kmeans::init::randomCSR> init(nClusters);

    init.input.set(kmeans::init::data, dataTable);
    init.compute();

    NumericTablePtr centroids = init.getResult()->get(kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm to calculate only assignments */
    kmeans::Batch<algorithmFPType, kmeans::lloydCSR> algorithm(nClusters, 0);

    algorithm.input.set(kmeans::data, dataTable);
    algorithm.input.set(kmeans::inputCentroids, centroids);

    algorithm.parameter().resultsToEvaluate = kmeans::computeAssignments;

    algorithm.compute();

    /* Print the clusterization results */
    printNumericTable(algorithm.getResult()->get(kmeans::assignments),
                      "First 10 cluster assignments:",
                      10);

    return 0;
}
