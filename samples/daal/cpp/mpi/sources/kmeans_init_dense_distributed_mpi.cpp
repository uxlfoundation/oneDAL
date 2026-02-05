/* file: kmeans_init_dense_distributed_mpi.cpp */
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
* <a name="DAAL-SAMPLE-CPP-KMEANS_INIT_DENSE_DISTRIBUTED"></a>
* \example kmeans_init_dense_distributed_mpi.cpp
*/

#include <mpi.h>
#include "daal.h"
#include "service.h"
#include "stdio.h"
#include <iostream>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

typedef std::vector<byte> ByteBuffer;
typedef float algorithmFPType; /* Algorithm floating-point type */

/* K-Means algorithm parameters */
const size_t nClusters = 20;
const size_t nIterations = 5;
const size_t nBlocks = 4;

/* Input data set parameters */
const std::string datasetFileName = "data/kmeans_dense.csv";

#define mpi_root 0
const int step3ResultSizeTag = 1;
const int step3ResultTag = 2;

NumericTablePtr loadData(int rankId) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();
    return dataSource.getNumericTable();
}

template <kmeans::init::Method method>
NumericTablePtr initCentroids(int rankId, const NumericTablePtr& pData);
NumericTablePtr computeCentroids(int rankId,
                                 const NumericTablePtr& pData,
                                 const NumericTablePtr& initialCentroids);

template <kmeans::init::Method method>
void runKMeans(int rankId, const NumericTablePtr& pData, const char* methodName) {
    if (rankId == mpi_root)
        std::cout << "K-means init parameters: method = " << methodName << std::endl;
    NumericTablePtr centroids = initCentroids<method>(rankId, pData);
    for (size_t it = 0; it < nIterations; it++)
        centroids = computeCentroids(rankId, pData, centroids);
    /* Print the clusterization results */
    if (rankId == mpi_root)
        printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10);
}

int main(int argc, char* argv[]) {
    int rankId, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    NumericTablePtr pData = loadData(rankId);

    runKMeans<kmeans::init::plusPlusDense>(rankId, pData, "plusPlusDense");
    runKMeans<kmeans::init::parallelPlusDense>(rankId, pData, "parallelPlusDense");

    MPI_Finalize();
    return 0;
}

static int lengthsToShifts(const int lengths[nBlocks], int shifts[nBlocks]) {
    int shift = 0;
    for (size_t i = 0; i < nBlocks; shift += lengths[i], ++i)
        shifts[i] = shift;
    return shift;
}

/* Send the value to all processes in the group and collect received values into one table */
static NumericTablePtr allToAll(const NumericTablePtr& value) {
    std::vector<NumericTablePtr> aRes;
    ByteBuffer dataToSend;
    if (value.get())
        serializeDAALObject(value.get(), dataToSend);
    const int dataToSendLength = dataToSend.size();
    int perNodeArchLength[nBlocks];
    for (size_t i = 0; i < nBlocks; i++)
        perNodeArchLength[i] = 0;

    MPI_Allgather(&dataToSendLength,
                  sizeof(int),
                  MPI_CHAR,
                  perNodeArchLength,
                  sizeof(int),
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    int perNodeArchShift[nBlocks];
    const int totalToReceive = lengthsToShifts(perNodeArchLength, perNodeArchShift);
    if (!totalToReceive)
        return NumericTablePtr();

    ByteBuffer dataToReceive(totalToReceive);
    MPI_Allgatherv(&dataToSend[0],
                   dataToSendLength,
                   MPI_CHAR,
                   &dataToReceive[0],
                   perNodeArchLength,
                   perNodeArchShift,
                   MPI_CHAR,
                   MPI_COMM_WORLD);

    for (size_t i = 0, shift = 0; i < nBlocks; shift += perNodeArchLength[i], ++i) {
        if (!perNodeArchLength[i])
            continue;
        NumericTablePtr pTbl =
            NumericTable::cast(deserializeDAALObject(&dataToReceive[shift], perNodeArchLength[i]));
        aRes.push_back(pTbl);
    }
    if (!aRes.size())
        return NumericTablePtr();
    if (aRes.size() == 1)
        return aRes[0];

    /* For parallelPlus algorithm */
    RowMergedNumericTablePtr pMerged(new RowMergedNumericTable());
    for (size_t i = 0; i < aRes.size(); ++i)
        pMerged->addNumericTable(aRes[i]);
    return NumericTable::cast(pMerged);
}

/* Send the value to all processes in the group and collect received values into one table */
static void allToMaster(int rankId,
                        const NumericTablePtr& value,
                        std::vector<NumericTablePtr>& aRes) {
    const bool isRoot = (rankId == mpi_root);
    aRes.clear();
    ByteBuffer dataToSend;
    if (value.get())
        serializeDAALObject(value.get(), dataToSend);
    const int dataToSendLength = dataToSend.size();
    int perNodeArchLength[nBlocks];
    for (size_t i = 0; i < nBlocks; i++)
        perNodeArchLength[i] = 0;

    MPI_Gather(&dataToSendLength,
               sizeof(int),
               MPI_CHAR,
               isRoot ? perNodeArchLength : NULL,
               sizeof(int),
               MPI_CHAR,
               mpi_root,
               MPI_COMM_WORLD);

    ByteBuffer dataToReceive;
    int perNodeArchShift[nBlocks];
    if (isRoot) {
        const int totalToReceive = lengthsToShifts(perNodeArchLength, perNodeArchShift);
        if (!totalToReceive)
            return;
        dataToReceive.resize(totalToReceive);
    }
    MPI_Gatherv(&dataToSend[0],
                dataToSendLength,
                MPI_CHAR,
                isRoot ? &dataToReceive[0] : NULL,
                perNodeArchLength,
                perNodeArchShift,
                MPI_CHAR,
                mpi_root,
                MPI_COMM_WORLD);

    if (!isRoot)
        return;
    aRes.resize(nBlocks);
    for (size_t i = 0, shift = 0; i < nBlocks; shift += perNodeArchLength[i], ++i) {
        if (perNodeArchLength[i])
            aRes[i] = NumericTable::cast(
                deserializeDAALObject(&dataToReceive[shift], perNodeArchLength[i]));
    }
}

template <kmeans::init::Method method>
NumericTablePtr initStep1(int rankId, const NumericTablePtr& pData) {
    const size_t nVectorsInBlock = pData->getNumberOfRows();
    /* Create an algorithm object for the K-Means algorithm */
    kmeans::init::Distributed<step1Local, algorithmFPType, method> local(nClusters,
                                                                         nBlocks * nVectorsInBlock,
                                                                         rankId * nVectorsInBlock);
    local.input.set(kmeans::init::data, pData);
    local.compute();
    return allToAll(local.getPartialResult()->get(kmeans::init::partialCentroids));
}

template <kmeans::init::Method method>
void initStep2(int rankId,
               const NumericTablePtr& pData,
               DataCollectionPtr& localNodeData,
               const NumericTablePtr& step2Input,
               bool bFirstIteration,
               std::vector<NumericTablePtr>& step2Results,
               bool bOutputForStep5Required = false) {
    kmeans::init::Distributed<step2Local, algorithmFPType, method> step2(nClusters,
                                                                         bFirstIteration);
    step2.parameter.outputForStep5Required = bOutputForStep5Required;
    step2.input.set(kmeans::init::data, pData);
    step2.input.set(kmeans::init::internalInput, localNodeData);
    step2.input.set(kmeans::init::inputOfStep2, step2Input);
    step2.compute();
    if (bFirstIteration)
        localNodeData = step2.getPartialResult()->get(kmeans::init::internalResult);
    allToMaster(rankId,
                step2.getPartialResult()->get(bOutputForStep5Required
                                                  ? kmeans::init::outputOfStep2ForStep5
                                                  : kmeans::init::outputOfStep2ForStep3),
                step2Results);
}

template <kmeans::init::Method method>
NumericTablePtr initStep3(kmeans::init::Distributed<step3Master, algorithmFPType, method>& step3,
                          std::vector<NumericTablePtr>& step2Results) {
    for (size_t i = 0; i < step2Results.size(); ++i)
        step3.input.add(kmeans::init::inputOfStep3FromStep2, i, step2Results[i]);
    step3.compute();
    ByteBuffer buff;
    NumericTablePtr step4InputOnRoot;
    for (size_t i = 0; i < nBlocks; ++i) {
        NumericTablePtr pTbl =
            step3.getPartialResult()->get(kmeans::init::outputOfStep3ForStep4, i); /* can be null */
        if (i == mpi_root) {
            step4InputOnRoot = pTbl;
            continue;
        }
        buff.clear();
        size_t size = pTbl.get() ? serializeDAALObject(pTbl.get(), buff) : 0;
        MPI_Send(&size, sizeof(size_t), MPI_BYTE, int(i), step3ResultSizeTag, MPI_COMM_WORLD);
        if (size)
            MPI_Send(&buff[0], size, MPI_BYTE, int(i), step3ResultTag, MPI_COMM_WORLD);
    }
    return step4InputOnRoot;
}

NumericTablePtr receiveStep3Output(int rankId) {
    size_t size = 0;
    MPI_Status status;
    MPI_Recv(&size,
             sizeof(size_t),
             MPI_BYTE,
             mpi_root,
             step3ResultSizeTag,
             MPI_COMM_WORLD,
             &status);
    if (size) {
        ByteBuffer buff(size);
        MPI_Recv(&buff[0], size, MPI_BYTE, mpi_root, step3ResultTag, MPI_COMM_WORLD, &status);
        return NumericTable::cast(deserializeDAALObject(&buff[0], size));
    }
    return NumericTablePtr();
}

template <kmeans::init::Method method>
NumericTablePtr initStep4(int rankId,
                          const NumericTablePtr& pData,
                          const DataCollectionPtr& localNodeData,
                          const NumericTablePtr& step4Input) {
    NumericTablePtr step4Result;
    if (step4Input) {
        /* Create an algorithm object for the step 4 */
        kmeans::init::Distributed<step4Local, algorithmFPType, method> step4(nClusters);
        /* Set the input data to the algorithm */
        step4.input.set(kmeans::init::data, pData);
        step4.input.set(kmeans::init::internalInput, localNodeData);
        step4.input.set(kmeans::init::inputOfStep4FromStep3, step4Input);
        /* Compute and get the result */
        step4.compute();
        step4Result = step4.getPartialResult()->get(kmeans::init::outputOfStep4);
    }
    return allToAll(step4Result);
}

template <>
NumericTablePtr initCentroids<kmeans::init::plusPlusDense>(int rankId,
                                                           const NumericTablePtr& pData) {
    const bool isRoot = (rankId == mpi_root);
    const kmeans::init::Method method = kmeans::init::plusPlusDense;
    /* Internal data to be stored on the local nodes */
    DataCollectionPtr localNodeData;
    /* Numeric table to collect the results */
    RowMergedNumericTablePtr pCentroids(new RowMergedNumericTable());
    /* First step on the local nodes */
    NumericTablePtr step2Input = initStep1<method>(rankId, pData);
    pCentroids->addNumericTable(step2Input);
    /* Create an algorithm object for the step 3 */
    typedef kmeans::init::Distributed<step3Master, algorithmFPType, method> Step3Master;
    SharedPtr<Step3Master> step3(isRoot ? new Step3Master(nClusters) : NULL);
    for (size_t iCenter = 1; iCenter < nClusters; ++iCenter) {
        std::vector<NumericTablePtr> step2ResultsOnMaster;
        initStep2<method>(rankId,
                          pData,
                          localNodeData,
                          step2Input,
                          iCenter == 1,
                          step2ResultsOnMaster);
        NumericTablePtr step4Input =
            (step3 ? initStep3<method>(*step3, step2ResultsOnMaster) : receiveStep3Output(rankId));
        step2Input = initStep4<method>(rankId, pData, localNodeData, step4Input);
        pCentroids->addNumericTable(step2Input);
    }
    return daal::data_management::convertToHomogen<float>(
        *pCentroids); /* can be returned as pCentroids as well */
}

template <>
NumericTablePtr initCentroids<kmeans::init::parallelPlusDense>(int rankId,
                                                               const NumericTablePtr& pData) {
    const bool isRoot = (rankId == mpi_root);
    const kmeans::init::Method method = kmeans::init::parallelPlusDense;
    /* default value of nRounds used by all steps */
    const size_t nRounds = kmeans::init::Parameter(nClusters).nRounds;

    /* Create an algorithm object for the step 5 */
    typedef kmeans::init::Distributed<step5Master, algorithmFPType, method> Step5Master;
    SharedPtr<Step5Master> step5(isRoot ? new Step5Master(nClusters) : NULL);

    /* Internal data to be stored on the local nodes */
    DataCollectionPtr localNodeData;

    /* First step on the local nodes */
    NumericTablePtr step2Input = initStep1<method>(rankId, pData);
    if (step5)
        step5->input.add(kmeans::init::inputCentroids, step2Input);

    /* Create an algorithm object for the step 3 */
    typedef kmeans::init::Distributed<step3Master, algorithmFPType, method> Step3Master;
    SharedPtr<Step3Master> step3(isRoot ? new Step3Master(nClusters) : NULL);
    for (size_t iRound = 0; iRound < nRounds; ++iRound) {
        /* Perform step 2 */
        std::vector<NumericTablePtr> step2ResultsOnMaster;
        initStep2<method>(rankId,
                          pData,
                          localNodeData,
                          step2Input,
                          iRound == 0,
                          step2ResultsOnMaster);
        /* Perform step 3 */
        NumericTablePtr step4Input =
            (step3 ? initStep3<method>(*step3, step2ResultsOnMaster) : receiveStep3Output(rankId));
        /* Perform step 4 */
        step2Input = initStep4<method>(rankId, pData, localNodeData, step4Input);
        if (step5)
            step5->input.add(kmeans::init::inputCentroids, step2Input);
    }

    /* One more step 2 */
    std::vector<NumericTablePtr> step2Results;
    initStep2<method>(rankId, pData, localNodeData, step2Input, false, step2Results, true);
    if (step5) /* isRoot == true */
    {
        for (size_t i = 0; i < step2Results.size(); ++i)
            step5->input.add(kmeans::init::inputOfStep5FromStep2, step2Results[i]);
        step5->input.set(kmeans::init::inputOfStep5FromStep3,
                         step3->getPartialResult()->get(kmeans::init::outputOfStep3ForStep5));
        step5->compute();
        step5->finalizeCompute();
        return step5->getResult()->get(kmeans::init::centroids);
    }
    return NumericTablePtr();
}

NumericTablePtr computeCentroids(int rankId,
                                 const NumericTablePtr& pData,
                                 const NumericTablePtr& initialCentroids) {
    const bool isRoot = (rankId == mpi_root);
    ByteBuffer nodeCentroids;
    size_t CentroidsArchLength =
        (isRoot ? serializeDAALObject(initialCentroids.get(), nodeCentroids) : 0);

    /* Get centroids from the root node */
    MPI_Bcast(&CentroidsArchLength, sizeof(size_t), MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    if (!isRoot)
        nodeCentroids.resize(CentroidsArchLength);
    MPI_Bcast(&nodeCentroids[0], CentroidsArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD);

    NumericTablePtr centroids =
        NumericTable::cast(deserializeDAALObject(&nodeCentroids[0], CentroidsArchLength));

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::Distributed<step1Local, algorithmFPType, kmeans::lloydDense> localAlgorithm(nClusters);

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(kmeans::data, pData);
    localAlgorithm.input.set(kmeans::inputCentroids, centroids);

    /* Compute k-means */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    ByteBuffer nodeResults;
    size_t perNodeArchLength =
        serializeDAALObject(localAlgorithm.getPartialResult().get(), nodeResults);

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    ByteBuffer serializedData;
    if (isRoot)
        serializedData.resize(perNodeArchLength * nBlocks);

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
        kmeans::Distributed<step2Master, algorithmFPType, kmeans::lloydDense> masterAlgorithm(
            nClusters);

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            SerializationIfacePtr ptr =
                deserializeDAALObject(&serializedData[perNodeArchLength * i], perNodeArchLength);
            kmeans::PartialResultPtr dataForStep2FromStep1 =
                dynamicPointerCast<kmeans::PartialResult, SerializationIface>(ptr);

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
