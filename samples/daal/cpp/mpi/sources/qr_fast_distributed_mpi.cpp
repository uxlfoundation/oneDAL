/* file: qr_fast_distributed_mpi.cpp */
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
!    C++ sample of computing QR decomposition in the distributed processing
!    mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-QR_FAST_DISTRIBUTED_MPI"></a>
 * \example qr_fast_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileName = "data/qr.csv";

void computestep1Local();
void computeOnMasterNode();
void finalizeComputestep1Local();

int rankId;
int commSize;
#define mpi_root 0

data_management::DataCollectionPtr dataFromStep1ForStep3;
NumericTablePtr R;
NumericTablePtr Qi;

services::SharedPtr<byte> serializedData;
size_t perNodeArchLength;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    if (nBlocks != commSize) {
        if (rankId == mpiRoot) {
            std::cout << commSize << " MPI ranks != " << nBlocks
                      << " datasets available, so please start exactly " << nBlocks << " ranks."
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    computestep1Local();

    if (rankId == mpiRoot) {
        computeOnMasterNode();
    }

    finalizeComputestep1Local();

    /* Print the results */
    if (rankId == mpiRoot) {
        printNumericTable(Qi, "Part of orthogonal matrix Q from 1st node:", 10);
        printNumericTable(R, "Triangular matrix R:");
    }

    MPI_Finalize();

    return 0;
}

void computestep1Local() {
    /* 1. Count total rows (only root needed, broadcast later) */
    size_t totalRows = 0;
    if (rankId == mpi_root) {
        totalRows = countRowsCSV(datasetFileName);
    }
    MPI_Bcast(&totalRows, 1, MPI_UNSIGNED_LONG, mpi_root, MPI_COMM_WORLD);

    /* 2. Compute block size for each process */
    size_t blockSize = (totalRows + comm_size - 1) / comm_size;
    size_t rowOffset = rankId * blockSize;
    size_t rowsToRead = std::min(blockSize, totalRows - rowOffset);

    /* 3. Each process reads only its block */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    if (rowsToRead > 0) {
        dataSource.loadDataBlock(rowsToRead, rowOffset, rowsToRead);
    }

    NumericTablePtr localData = dataSource.getNumericTable();

    /* Create an algorithm to compute QR decomposition on local nodes */
    qr::Distributed<step1Local> alg;

    alg.input.set(qr::data, localData);

    /* Compute QR decomposition */
    alg.compute();

    data_management::DataCollectionPtr dataFromStep1ForStep2;
    dataFromStep1ForStep2 = alg.getPartialResult()->get(qr::outputOfStep1ForStep2);
    dataFromStep1ForStep3 = alg.getPartialResult()->get(qr::outputOfStep1ForStep3);

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    dataFromStep1ForStep2->serialize(dataArch);
    perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (rankId == mpiRoot) {
        serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);
    }

    byte* nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */

    MPI_Gather(nodeResults,
               perNodeArchLength,
               MPI_CHAR,
               serializedData.get(),
               perNodeArchLength,
               MPI_CHAR,
               mpiRoot,
               MPI_COMM_WORLD);

    delete[] nodeResults;
}

void computeOnMasterNode() {
    /* Create an algorithm to compute QR decomposition on the master node */
    qr::Distributed<step2Master> alg;

    for (size_t i = 0; i < nBlocks; i++) {
        /* Deserialize partial results from step 1 */
        OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);

        data_management::DataCollectionPtr dataForStep2FromStep1 =
            data_management::DataCollectionPtr(new data_management::DataCollection());
        dataForStep2FromStep1->deserialize(dataArch);

        alg.input.add(qr::inputOfStep2FromStep1, i, dataForStep2FromStep1);
    }

    /* Compute QR decomposition */
    alg.compute();

    qr::DistributedPartialResultPtr pres = alg.getPartialResult();
    KeyValueDataCollectionPtr inputForStep3FromStep2 = pres->get(qr::outputOfStep2ForStep3);

    for (size_t i = 0; i < nBlocks; i++) {
        /* Serialize partial results to transfer to local nodes for step 3 */
        InputDataArchive dataArch;
        (*inputForStep3FromStep2)[i]->serialize(dataArch);

        if (i == 0) {
            perNodeArchLength = dataArch.getSizeOfArchive();
            /* Serialized data is of equal size for each node if it was equal in step 1 */
            serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);
        }

        dataArch.copyArchiveToArray(serializedData.get() + perNodeArchLength * i,
                                    perNodeArchLength);
    }

    qr::ResultPtr res = alg.getResult();

    R = res->get(qr::matrixR);
}

void finalizeComputestep1Local() {
    /* Get the size of the serialized input */
    MPI_Bcast(&perNodeArchLength, sizeof(size_t), MPI_CHAR, mpiRoot, MPI_COMM_WORLD);

    byte* nodeResults = new byte[perNodeArchLength];

    /* Transfer partial results from the root node */

    MPI_Scatter(serializedData.get(),
                perNodeArchLength,
                MPI_CHAR,
                nodeResults,
                perNodeArchLength,
                MPI_CHAR,
                mpiRoot,
                MPI_COMM_WORLD);

    /* Deserialize partial results from step 2 */
    OutputDataArchive dataArch(nodeResults, perNodeArchLength);

    data_management::DataCollectionPtr dataFromStep2ForStep3 =
        data_management::DataCollectionPtr(new data_management::DataCollection());
    dataFromStep2ForStep3->deserialize(dataArch);

    delete[] nodeResults;

    /* Create an algorithm to compute QR decomposition on the master node */
    qr::Distributed<step3Local> alg;

    alg.input.set(qr::inputOfStep3FromStep1, dataFromStep1ForStep3);
    alg.input.set(qr::inputOfStep3FromStep2, dataFromStep2ForStep3);

    /* Compute QR decomposition */
    alg.compute();
    alg.finalizeCompute();

    qr::ResultPtr res = alg.getResult();

    Qi = res->get(qr::matrixQ);
}
