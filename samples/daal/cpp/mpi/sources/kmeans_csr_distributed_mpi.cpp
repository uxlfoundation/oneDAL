/* file: kmeans_csr_distributed_mpi.cpp */
/*******************************************************************************
* Copyright 2017 Intel Corporation
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
!    C++ sample of K-Means clustering in the distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-KMEANS_CSR_DISTRIBUTED"></a>
 * \example kmeans_csr_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"
#include "stdio.h"

using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType; /* Algorithm floating-point type */

typedef std::vector<byte> ByteBuffer;

/* K-Means algorithm parameters */
const size_t nClusters = 20;
const size_t nIterations = 5;
const size_t nBlocks = 4;

/* Input data set parameters */
const std::string datasetFileName = "data/kmeans_csr.csv";

int rankId, comm_size;
#define mpi_root 0

NumericTablePtr loadData(int rankId) {
    return NumericTablePtr(createSparseTable<float>(datasetFileName));
}

NumericTablePtr init(int rankId, const NumericTablePtr& pData);
NumericTablePtr compute(int rankId,
                        const NumericTablePtr& pData,
                        const NumericTablePtr& initialCentroids);

int main(int argc, char* argv[]) {
    int rankId, comm_size;
    checkArguments(argc, argv, 1, &datasetFileName);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    NumericTablePtr pData = loadData(rankId);
    NumericTablePtr centroids = init(rankId, pData);

    for (size_t it = 0; it < nIterations; it++)
        centroids = compute(rankId, pData, centroids);

    /* Print the clusterization results */
    if (rankId == mpi_root)
        printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10);

    MPI_Finalize();
    return 0;
}

NumericTablePtr init(int rankId, const NumericTablePtr& pData) {
    const bool isRoot = (rankId == mpi_root);

    const size_t nVectorsInBlock = pData->getNumberOfRows();

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::init::Distributed<step1Local, algorithmFPType, kmeans::init::randomCSR> localInit(
        nClusters,
        nBlocks * nVectorsInBlock,
        rankId * nVectorsInBlock);

    /* Set the input data set to the algorithm */
    localInit.input.set(kmeans::init::data, pData);

    /* Compute k-means */
    localInit.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localInit.getPartialResult()->serialize(dataArch);
    const int perNodeArchLength = (int)dataArch.getSizeOfArchive();

    int aPerNodeArchLength[nBlocks];
    /* Transfer archive length to the step 2 on the root node */
    MPI_Gather(&perNodeArchLength,
               sizeof(int),
               MPI_CHAR,
               isRoot ? &aPerNodeArchLength[0] : NULL,
               sizeof(int),
               MPI_CHAR,
               mpi_root,
               MPI_COMM_WORLD);

    ByteBuffer serializedData;
    /* Calculate total archive length */
    int totalArchLength = 0;
    int displs[nBlocks];
    if (isRoot) {
        for (size_t i = 0; i < nBlocks; ++i) {
            displs[i] = totalArchLength;
            totalArchLength += aPerNodeArchLength[i];
        }
        serializedData.resize(totalArchLength);
    }

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    MPI_Gatherv(&nodeResults[0],
                perNodeArchLength,
                MPI_CHAR,
                serializedData.size() ? &serializedData[0] : NULL,
                aPerNodeArchLength,
                displs,
                MPI_CHAR,
                mpi_root,
                MPI_COMM_WORLD);

    if (isRoot) {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::init::Distributed<step2Master, algorithmFPType, kmeans::init::randomCSR> masterInit(
            nClusters);
        for (size_t i = 0, shift = 0; i < nBlocks; shift += aPerNodeArchLength[i], ++i) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[shift], aPerNodeArchLength[i]);

            kmeans::init::PartialResultPtr dataForStep2FromStep1(new kmeans::init::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterInit.input.add(kmeans::init::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterInit.compute();
        masterInit.finalizeCompute();

        return masterInit.getResult()->get(kmeans::init::centroids);
    }
    return NumericTablePtr();
}

NumericTablePtr compute(int rankId,
                        const NumericTablePtr& pData,
                        const NumericTablePtr& initialCentroids) {
    const bool isRoot = (rankId == mpi_root);
    size_t CentroidsArchLength = 0;
    InputDataArchive inputArch;
    if (isRoot) {
        /*Retrieve the algorithm results and serialize them */
        initialCentroids->serialize(inputArch);
        CentroidsArchLength = inputArch.getSizeOfArchive();
    }

    /* Get partial results from the root node */
    MPI_Bcast(&CentroidsArchLength, sizeof(size_t), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    ByteBuffer nodeCentroids(CentroidsArchLength);
    if (isRoot)
        inputArch.copyArchiveToArray(&nodeCentroids[0], CentroidsArchLength);

    MPI_Bcast(&nodeCentroids[0], CentroidsArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    /* Deserialize centroids data */
    OutputDataArchive outArch(nodeCentroids.size() ? &nodeCentroids[0] : NULL, CentroidsArchLength);

    NumericTablePtr centroids(new HomogenNumericTable<>());

    centroids->deserialize(outArch);

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::Distributed<step1Local, algorithmFPType, kmeans::lloydCSR> localAlgorithm(nClusters);

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(kmeans::data, pData);
    localAlgorithm.input.set(kmeans::inputCentroids, centroids);

    /* Compute k-means */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    ByteBuffer serializedData;

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (isRoot)
        serializedData.resize(perNodeArchLength * nBlocks);

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    MPI_Gather(&nodeResults[0],
               perNodeArchLength,
               MPI_CHAR,
               serializedData.size() ? &serializedData[0] : NULL,
               perNodeArchLength,
               MPI_CHAR,
               mpi_root,
               MPI_COMM_WORLD);

    if (isRoot) {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::Distributed<step2Master, algorithmFPType, kmeans::lloydCSR> masterAlgorithm(
            nClusters);

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[perNodeArchLength * i], perNodeArchLength);

            kmeans::PartialResultPtr dataForStep2FromStep1(new kmeans::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterAlgorithm.input.add(kmeans::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        return masterAlgorithm.getResult()->get(kmeans::centroids);
    }
    return NumericTablePtr();
}
