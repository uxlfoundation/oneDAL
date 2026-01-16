/* file: implicit_als_csr_distributed_mpi.cpp */
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
!    C++ example of the implicit alternating least squares (ALS) algorithm in
!    the distributed processing mode.
!
!    The program trains the implicit ALS model on a training data set.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-IMPLICIT_ALS_CSR_DISTRIBUTED"></a>
 * \example implicit_als_csr_distributed_mpi.cpp
 */

#include "mpi.h"
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms::implicit_als;

/* Input data set parameters */
const size_t nBlocks = 4;

int rankId, comm_size;
#define mpi_root 0

/* Number of observations in transposed training data set blocks */
const std::string trainDatasetFileNames[nBlocks] = { "data/implicit_als_trans_csr_1.csv",
                                                     "data/implicit_als_trans_csr_2.csv",
                                                     "data/implicit_als_trans_csr_3.csv",
                                                     "data/implicit_als_trans_csr_4.csv" };

static int usersPartition[1] = { nBlocks };

NumericTablePtr userOffset;
NumericTablePtr itemOffset;

KeyValueDataCollectionPtr userOffsetsOnMaster;
KeyValueDataCollectionPtr itemOffsetsOnMaster;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Algorithm parameters */
const size_t nUsers = 46; /* Full number of users */

const size_t nFactors = 2; /* Number of factors */
const size_t maxIterations = 5; /* Number of iterations in the implicit ALS training algorithm */

int displs[nBlocks];
int sdispls[nBlocks];
int rdispls[nBlocks];

std::string colFileName;

CSRNumericTablePtr dataTable;
CSRNumericTablePtr transposedDataTable;

KeyValueDataCollectionPtr userStep3LocalInput;
KeyValueDataCollectionPtr itemStep3LocalInput;

training::DistributedPartialResultStep4Ptr itemsPartialResultLocal;
training::DistributedPartialResultStep4Ptr usersPartialResultLocal;
training::DistributedPartialResultStep4Ptr itemsPartialResultsMaster[nBlocks];

NumericTablePtr predictedRatingsLocal[nBlocks];
NumericTablePtr predictedRatingsMaster[nBlocks][nBlocks];

ByteBuffer serializedData;
ByteBuffer serializedSendData;
ByteBuffer serializedRecvData;

void initializeModel();
void readData();
void trainModel();
void testModel();
void predictRatings();

template <typename T>
void gather(const ByteBuffer &nodeResults, T *result);
void gatherItems(const ByteBuffer &nodeResults);
template <typename T>
void all2all(ByteBuffer *nodeResults, KeyValueDataCollectionPtr result);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    readData();

    initializeModel();

    trainModel();

    testModel();

    if (rankId == mpi_root) {
        for (size_t i = 0; i < nBlocks; i++) {
            for (size_t j = 0; j < nBlocks; j++) {
                std::cout << "Ratings for users block " << i << ", items block " << j << " :"
                          << std::endl;
                printALSRatings(NumericTable::cast((*userOffsetsOnMaster)[i]),
                                NumericTable::cast((*itemOffsetsOnMaster)[j]),
                                predictedRatingsMaster[i][j]);
            }
        }
    }

    MPI_Finalize();
    return 0;
}

void readData() {
    /* Read trainDatasetFileName from a file and create a numeric table to store the input data */
    dataTable.reset(createSparseTable<float>(trainDatasetFileNames[rankId]));
}

KeyValueDataCollectionPtr initializeStep1Local() {
    /* Create an algorithm object to initialize the implicit ALS model with the default method */
    training::init::Distributed<step1Local, algorithmFPType, training::init::fastCSR> initAlgorithm;
    initAlgorithm.parameter.fullNUsers = nUsers;
    initAlgorithm.parameter.nFactors = nFactors;
    initAlgorithm.parameter.seed += rankId;
    initAlgorithm.parameter.partition.reset(
        new HomogenNumericTable<int>((int *)usersPartition, 1, 1));
    /* Pass a training data set and dependent values to the algorithm */
    initAlgorithm.input.set(training::init::data, dataTable);

    /* Initialize the implicit ALS model */
    initAlgorithm.compute();

    training::init::PartialResultPtr partialResult = initAlgorithm.getPartialResult();
    itemStep3LocalInput = partialResult->get(training::init::outputOfInitForComputeStep3);
    userOffset = partialResult->get(training::init::offsets, (size_t)rankId);
    if (rankId == mpi_root) {
        userOffsetsOnMaster = partialResult->get(training::init::offsets);
    }
    PartialModelPtr partialModelLocal = partialResult->get(training::init::partialModel);

    itemsPartialResultLocal.reset(new training::DistributedPartialResultStep4());
    itemsPartialResultLocal->set(training::outputOfStep4ForStep1, partialModelLocal);

    return partialResult->get(training::init::outputOfStep1ForStep2);
}

void initializeStep2Local(const KeyValueDataCollectionPtr &initStep2LocalInput) {
    /* Create an algorithm object to perform the second step of the implicit ALS initialization algorithm */
    training::init::Distributed<step2Local, algorithmFPType, training::init::fastCSR> initAlgorithm;

    initAlgorithm.input.set(training::init::inputOfStep2FromStep1, initStep2LocalInput);

    /* Compute partial results of the second step on local nodes */
    initAlgorithm.compute();

    training::init::DistributedPartialResultStep2Ptr partialResult =
        initAlgorithm.getPartialResult();
    transposedDataTable = CSRNumericTable::cast(partialResult->get(training::init::transposedData));
    userStep3LocalInput = partialResult->get(training::init::outputOfInitForComputeStep3);
    itemOffset = partialResult->get(training::init::offsets, (size_t)rankId);
    if (rankId == mpi_root) {
        itemOffsetsOnMaster = partialResult->get(training::init::offsets);
    }
}

void initializeModel() {
    KeyValueDataCollectionPtr initStep1LocalResult = initializeStep1Local();

    /* MPI_Alltoallv to populate initStep2LocalInput */
    ByteBuffer nodeCPs[nBlocks];
    for (size_t i = 0; i < nBlocks; i++) {
        serializeDAALObject((*initStep1LocalResult)[i].get(), nodeCPs[i]);
    }
    KeyValueDataCollectionPtr initStep2LocalInput(new KeyValueDataCollection());
    all2all<NumericTable>(nodeCPs, initStep2LocalInput);

    initializeStep2Local(initStep2LocalInput);
}

training::DistributedPartialResultStep1Ptr computeStep1Local(
    const training::DistributedPartialResultStep4Ptr &partialResultLocal) {
    /* Create algorithm objects to compute implicit ALS algorithm in the distributed processing mode on the local node using the default method */
    training::Distributed<step1Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModel,
                        partialResultLocal->get(training::outputOfStep4ForStep1));

    /* Compute partial estimates on local nodes */
    algorithm.compute();

    /* Get the computed partial estimates */
    return algorithm.getPartialResult();
}

NumericTablePtr computeStep2Master(
    const training::DistributedPartialResultStep1Ptr *step1LocalResultsOnMaster) {
    /* Create algorithm objects to compute implicit ALS algorithm in the distributed processing mode on the master node using the default method */
    training::Distributed<step2Master> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set input objects for the algorithm */
    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(training::inputOfStep2FromStep1, step1LocalResultsOnMaster[i]);
    }

    /* Compute a partial estimate on the master node from the partial estimates on local nodes */
    algorithm.compute();

    return algorithm.getPartialResult()->get(training::outputOfStep2ForStep4);
}

KeyValueDataCollectionPtr computeStep3Local(
    const NumericTablePtr &offset,
    const training::DistributedPartialResultStep4Ptr &partialResultLocal,
    const KeyValueDataCollectionPtr &step3LocalInput) {
    training::Distributed<step3Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    algorithm.input.set(training::partialModel,
                        partialResultLocal->get(training::outputOfStep4ForStep3));
    algorithm.input.set(training::inputOfStep3FromInit, step3LocalInput);
    algorithm.input.set(training::offset, offset);

    algorithm.compute();

    return algorithm.getPartialResult()->get(training::outputOfStep3ForStep4);
}

training::DistributedPartialResultStep4Ptr computeStep4Local(
    const CSRNumericTablePtr &dataTable,
    const NumericTablePtr &step2MasterResult,
    const KeyValueDataCollectionPtr &step4LocalInput) {
    training::Distributed<step4Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    algorithm.input.set(training::partialModels, step4LocalInput);
    algorithm.input.set(training::partialData, dataTable);
    algorithm.input.set(training::inputOfStep4FromStep2, step2MasterResult);

    algorithm.compute();

    return algorithm.getPartialResult();
}

void trainModel() {
    training::DistributedPartialResultStep1Ptr step1LocalResultsOnMaster[nBlocks];
    training::DistributedPartialResultStep1Ptr step1LocalResult;
    NumericTablePtr step2MasterResult;
    KeyValueDataCollectionPtr step3LocalResult;
    KeyValueDataCollectionPtr step4LocalInput(new KeyValueDataCollection());

    ByteBuffer nodeCPs[nBlocks];
    ByteBuffer nodeResults;
    ByteBuffer crossProductBuf;
    int crossProductLen;

    for (size_t iteration = 0; iteration < maxIterations; iteration++) {
        step1LocalResult = computeStep1Local(itemsPartialResultLocal);

        serializeDAALObject(step1LocalResult.get(), nodeResults);

        /* Gathering step1LocalResult on the master */
        gather(nodeResults, step1LocalResultsOnMaster);

        if (rankId == mpi_root) {
            step2MasterResult = computeStep2Master(step1LocalResultsOnMaster);
            serializeDAALObject(step2MasterResult.get(), crossProductBuf);
            crossProductLen = crossProductBuf.size();
        }

        MPI_Bcast(&crossProductLen, sizeof(int), MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        if (rankId != mpi_root) {
            crossProductBuf.resize(crossProductLen);
        }
        MPI_Bcast(&crossProductBuf[0], crossProductLen, MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        step2MasterResult =
            NumericTable::cast(deserializeDAALObject(&crossProductBuf[0], crossProductLen));

        step3LocalResult =
            computeStep3Local(itemOffset, itemsPartialResultLocal, itemStep3LocalInput);

        /* MPI_Alltoallv to populate step4LocalInput */
        for (size_t i = 0; i < nBlocks; i++) {
            serializeDAALObject((*step3LocalResult)[i].get(), nodeCPs[i]);
        }
        all2all<PartialModel>(nodeCPs, step4LocalInput);

        usersPartialResultLocal =
            computeStep4Local(transposedDataTable, step2MasterResult, step4LocalInput);

        step1LocalResult = computeStep1Local(usersPartialResultLocal);

        serializeDAALObject(step1LocalResult.get(), nodeResults);

        /*Gathering step1LocalResult on the master*/
        gather(nodeResults, step1LocalResultsOnMaster);

        if (rankId == mpi_root) {
            step2MasterResult = computeStep2Master(step1LocalResultsOnMaster);
            serializeDAALObject(step2MasterResult.get(), crossProductBuf);
            crossProductLen = crossProductBuf.size();
        }

        MPI_Bcast(&crossProductLen, sizeof(int), MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        if (rankId != mpi_root) {
            crossProductBuf.resize(crossProductLen);
        }
        MPI_Bcast(&crossProductBuf[0], crossProductLen, MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        step2MasterResult =
            NumericTable::cast(deserializeDAALObject(&crossProductBuf[0], crossProductLen));

        step3LocalResult =
            computeStep3Local(userOffset, usersPartialResultLocal, userStep3LocalInput);

        /* MPI_Alltoallv to populate step4LocalInput */
        for (size_t i = 0; i < nBlocks; i++) {
            serializeDAALObject((*step3LocalResult)[i].get(), nodeCPs[i]);
        }
        all2all<PartialModel>(nodeCPs, step4LocalInput);

        itemsPartialResultLocal = computeStep4Local(dataTable, step2MasterResult, step4LocalInput);
    }

    /*Gather all itemsPartialResultLocal to itemsPartialResultsMaster on the master and distributing the result over other ranks*/
    serializeDAALObject(itemsPartialResultLocal.get(), nodeResults);
    gatherItems(nodeResults);
}

void testModel() {
    ByteBuffer nodeResults;
    /* Create an algorithm object to predict recommendations of the implicit ALS model */
    for (size_t i = 0; i < nBlocks; i++) {
        prediction::ratings::Distributed<step1Local> algorithm;
        algorithm.parameter.nFactors = nFactors;

        algorithm.input.set(prediction::ratings::usersPartialModel,
                            usersPartialResultLocal->get(training::outputOfStep4ForStep1));
        algorithm.input.set(prediction::ratings::itemsPartialModel,
                            itemsPartialResultsMaster[i]->get(training::outputOfStep4ForStep1));

        algorithm.compute();

        predictedRatingsLocal[i] = algorithm.getResult()->get(prediction::ratings::prediction);

        serializeDAALObject(predictedRatingsLocal[i].get(), nodeResults);
        gather(nodeResults, predictedRatingsMaster[i]);
    }
}

template <typename T>
void gather(const ByteBuffer &nodeResults, T *result) {
    int perNodeArchLengthMaster[nBlocks];
    int perNodeArchLength = nodeResults.size();
    MPI_Gather(&perNodeArchLength,
               sizeof(int),
               MPI_CHAR,
               perNodeArchLengthMaster,
               sizeof(int),
               MPI_CHAR,
               mpi_root,
               MPI_COMM_WORLD);

    if (rankId == mpi_root) {
        int memoryBuf = 0;
        for (size_t i = 0; i < nBlocks; i++) {
            memoryBuf += perNodeArchLengthMaster[i];
        }
        serializedData.resize(memoryBuf);

        size_t shift = 0;
        for (size_t i = 0; i < nBlocks; i++) {
            displs[i] = shift;
            shift += perNodeArchLengthMaster[i];
        }
    }

    /* Transfer partial results to step 2 on the root node */
    MPI_Gatherv(&nodeResults[0],
                perNodeArchLength,
                MPI_CHAR,
                &serializedData[0],
                perNodeArchLengthMaster,
                displs,
                MPI_CHAR,
                mpi_root,
                MPI_COMM_WORLD);

    if (rankId == mpi_root) {
        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            result[i] = result[i]->cast(
                deserializeDAALObject(&serializedData[0] + displs[i], perNodeArchLengthMaster[i]));
        }
    }
}

void gatherItems(const ByteBuffer &nodeResults) {
    int perNodeArchLengthMaster[nBlocks];
    int perNodeArchLength = nodeResults.size();
    MPI_Allgather(&perNodeArchLength,
                  sizeof(int),
                  MPI_CHAR,
                  perNodeArchLengthMaster,
                  sizeof(int),
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    int memoryBuf = 0;
    for (int i = 0; i < nBlocks; i++) {
        memoryBuf += perNodeArchLengthMaster[i];
    }
    serializedData.resize(memoryBuf);

    size_t shift = 0;
    for (size_t i = 0; i < nBlocks; i++) {
        displs[i] = shift;
        shift += perNodeArchLengthMaster[i];
    }

    /* Transfer partial results to step 2 on the root node */
    MPI_Allgatherv(&nodeResults[0],
                   perNodeArchLength,
                   MPI_CHAR,
                   &serializedData[0],
                   perNodeArchLengthMaster,
                   displs,
                   MPI_CHAR,
                   MPI_COMM_WORLD);

    for (size_t i = 0; i < nBlocks; i++) {
        /* Deserialize partial results from step 4 */
        itemsPartialResultsMaster[i] = training::DistributedPartialResultStep4::cast(
            deserializeDAALObject(&serializedData[0] + displs[i], perNodeArchLengthMaster[i]));
    }
}

template <typename T>
void all2all(ByteBuffer *nodeResults, KeyValueDataCollectionPtr result) {
    int memoryBuf = 0;
    size_t shift = 0;
    int perNodeArchLengths[nBlocks];
    int perNodeArchLengthsRecv[nBlocks];
    for (int i = 0; i < nBlocks; i++) {
        perNodeArchLengths[i] = nodeResults[i].size();
        memoryBuf += perNodeArchLengths[i];
        sdispls[i] = shift;
        shift += perNodeArchLengths[i];
    }
    serializedSendData.resize(memoryBuf);

    /* memcpy to avoid double compute */
    memoryBuf = 0;
    for (int i = 0; i < nBlocks; i++) {
        for (int j = 0; j < perNodeArchLengths[i]; j++)
            serializedSendData[memoryBuf + j] = nodeResults[i][j];
        memoryBuf += perNodeArchLengths[i];
    }

    MPI_Alltoall(perNodeArchLengths,
                 sizeof(int),
                 MPI_CHAR,
                 perNodeArchLengthsRecv,
                 sizeof(int),
                 MPI_CHAR,
                 MPI_COMM_WORLD);

    memoryBuf = 0;
    shift = 0;
    for (int i = 0; i < nBlocks; i++) {
        memoryBuf += perNodeArchLengthsRecv[i];
        rdispls[i] = shift;
        shift += perNodeArchLengthsRecv[i];
    }
    serializedRecvData.resize(memoryBuf);

    /* Transfer partial results to step 2 on the root node */
    MPI_Alltoallv(&serializedSendData[0],
                  perNodeArchLengths,
                  sdispls,
                  MPI_CHAR,
                  &serializedRecvData[0],
                  perNodeArchLengthsRecv,
                  rdispls,
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    for (size_t i = 0; i < nBlocks; i++) {
        (*result)[i] = T::cast(
            deserializeDAALObject(&serializedRecvData[rdispls[i]], perNodeArchLengthsRecv[i]));
    }
}
