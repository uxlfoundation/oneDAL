/* file: kmeans_csr_distr.cpp */
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
!    C++ example of sparse K-Means clustering in the distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KMEANS_CSR_DISTRIBUTED"></a>
 * \example kmeans_csr_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* K-Means algorithm parameters */
const size_t nClusters = 20;
const size_t nIterations = 5;
const size_t nBlocks = 4;

const std::string datasetFileName = { "data/kmeans_csr.csv" };

CSRNumericTablePtr dataTable[nBlocks];
int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    CSRNumericTablePtr fullData(createSparseTable<float>(datasetFileName));
    const size_t totalRows = fullData->getNumberOfRows();

    const size_t rowsPerBlock = (totalRows + nBlocks - 1) / nBlocks;

    for (size_t i = 0; i < nBlocks; ++i) {
        size_t rowStart = i * rowsPerBlock;
        size_t rowEnd = std::min(rowStart + rowsPerBlock, totalRows);
        dataTable[i] = splitCSRBlock<algorithmFPType>(fullData, rowStart, rowEnd);
    }

    kmeans::Distributed<step2Master, algorithmFPType, kmeans::lloydCSR> masterAlgorithm(nClusters);

    NumericTablePtr centroids;
    NumericTablePtr assignments[nBlocks];
    NumericTablePtr objectiveFunction;

    kmeans::init::Distributed<step2Master, algorithmFPType, kmeans::init::randomCSR> masterInit(
        nClusters);

    for (size_t i = 0; i < nBlocks; ++i) {
        if (!dataTable[i])
            continue;

        kmeans::init::Distributed<step1Local, algorithmFPType, kmeans::init::randomCSR> localInit(
            nClusters,
            totalRows,
            i * rowsPerBlock);

        localInit.input.set(kmeans::init::data, dataTable[i]);
        localInit.compute();

        masterInit.input.add(kmeans::init::partialResults, localInit.getPartialResult());
    }
    masterInit.compute();
    masterInit.finalizeCompute();
    centroids = masterInit.getResult()->get(kmeans::init::centroids);

    for (size_t it = 0; it < nIterations; ++it) {
        for (size_t i = 0; i < nBlocks; ++i) {
            if (!dataTable[i])
                continue;

            kmeans::Distributed<step1Local, algorithmFPType, kmeans::lloydCSR> localAlgorithm(
                nClusters,
                false);

            localAlgorithm.input.set(kmeans::data, dataTable[i]);
            localAlgorithm.input.set(kmeans::inputCentroids, centroids);

            localAlgorithm.compute();

            masterAlgorithm.input.add(kmeans::partialResults, localAlgorithm.getPartialResult());
        }

        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        centroids = masterAlgorithm.getResult()->get(kmeans::centroids);
        objectiveFunction = masterAlgorithm.getResult()->get(kmeans::objectiveFunction);
    }

    for (size_t i = 0; i < nBlocks; ++i) {
        kmeans::Batch<algorithmFPType, kmeans::lloydCSR> localAlgorithm(nClusters, 0);

        /* Set the input data to the algorithm */
        localAlgorithm.input.set(kmeans::data, dataTable[i]);
        localAlgorithm.input.set(kmeans::inputCentroids, centroids);

        localAlgorithm.compute();

        assignments[i] = localAlgorithm.getResult()->get(kmeans::assignments);
    }

    /* Print the clusterization results */
    printNumericTable(assignments[0], "First 10 cluster assignments from 1st node:", 10);
    printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(objectiveFunction, "Objective function value:");

    return 0;
}
