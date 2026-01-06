/* file: svd_fast_distributed_mpi.cpp */
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
!    C++ sample of computing singular value decomposition (SVD) in the
!    distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVD_FAST_DISTRIBUTED_MPI"></a>
 * \example svd_fast_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileNames[] = { "data/svd_1.csv",
                                         "data/svd_2.csv",
                                         "data/svd_3.csv",
                                         "data/svd_4.csv" };

void computestep1Local();
void computeOnMasterNode();
void finalizeComputestep1Local();

int rankId;
int commSize;
#define mpiRoot 0

data_management::DataCollectionPtr dataFromStep1ForStep3;
NumericTablePtr Sigma;
NumericTablePtr V;
NumericTablePtr Ui;

services::SharedPtr<byte> serializedData;
size_t perNodeArchLength;

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    if (nBlocks != commSize) {
        if (rankId == mpiRoot) {
            std::cout << commSize << " MPI ranks != " << nBlocks
                      << " datasets available, so please start exactly " << nBlocks << " ranks.\n"
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
        printNumericTable(Sigma, "Singular values:");
        printNumericTable(V, "Right orthogonal matrix V:");
        printNumericTable(Ui, "Part of left orthogonal matrix U from root node:", 10);
    }

    MPI_Finalize();

    return 0;
}

void computestep1Local() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileNames[rankId],
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the input data */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute SVD on local nodes */
    svd::Distributed<step1Local> alg;

    alg.input.set(svd::data, dataSource.getNumericTable());

    /* Compute SVD */
    alg.compute();

    data_management::DataCollectionPtr dataFromStep1ForStep2;
    dataFromStep1ForStep2 = alg.getPartialResult()->get(svd::outputOfStep1ForStep2);
    dataFromStep1ForStep3 = alg.getPartialResult()->get(svd::outputOfStep1ForStep3);

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
    /* Create an algorithm to compute SVD on the master node */
    svd::Distributed<step2Master> alg;

    for (size_t i = 0; i < nBlocks; i++) {
        /* Deserialize partial results from step 1 */
        OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);

        data_management::DataCollectionPtr dataForStep2FromStep1 =
            data_management::DataCollectionPtr(new data_management::DataCollection());
        dataForStep2FromStep1->deserialize(dataArch);

        alg.input.add(svd::inputOfStep2FromStep1, i, dataForStep2FromStep1);
    }

    /* Compute SVD */
    alg.compute();

    svd::DistributedPartialResultPtr pres = alg.getPartialResult();
    KeyValueDataCollectionPtr inputForStep3FromStep2 = pres->get(svd::outputOfStep2ForStep3);

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

    svd::ResultPtr res = alg.getResult();

    Sigma = res->get(svd::singularValues);
    V = res->get(svd::rightSingularMatrix);
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

    /* Create an algorithm to compute SVD on the master node */
    svd::Distributed<step3Local> alg;

    alg.input.set(svd::inputOfStep3FromStep1, dataFromStep1ForStep3);
    alg.input.set(svd::inputOfStep3FromStep2, dataFromStep2ForStep3);

    /* Compute SVD */
    alg.compute();
    alg.finalizeCompute();

    svd::ResultPtr res = alg.getResult();

    Ui = res->get(svd::leftSingularMatrix);
}
