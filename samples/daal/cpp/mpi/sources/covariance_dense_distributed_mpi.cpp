/* file: covariance_dense_distributed_mpi.cpp */
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
!    C++ sample of dense variance-covariance matrix computation in the
!    distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-COVARIANCE_DENSE_DISTRIBUTED"></a>
 * \example covariance_dense_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;

const std::string datasetFileName = "data/covcormoments_dense.csv";
#define mpi_root 0

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rankId, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    checkArguments(argc, argv, 1, &datasetFileName);

    size_t totalRows = 0;
    if (rankId == mpi_root) {
        totalRows = countRowsCSV(datasetFileName);
    }
    MPI_Bcast(&totalRows, 1, MPI_UNSIGNED_LONG, mpi_root, MPI_COMM_WORLD);

    size_t blockSize = (totalRows + comm_size - 1) / comm_size;
    size_t rowOffset = rankId * blockSize;

    size_t rowsToRead = 0;
    if (rowOffset < totalRows) {
        rowsToRead = std::min(blockSize, totalRows - rowOffset);
    }

    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    size_t skipRows = rowOffset;
    while (skipRows > 0) {
        size_t skipped = dataSource.loadDataBlock(skipRows);
        if (skipped == 0)
            break;
        skipRows -= skipped;
    }

    NumericTablePtr localData;

    dataSource.loadDataBlock(rowsToRead);
    localData = dataSource.getNumericTable();

    covariance::Distributed<step1Local> localAlgorithm;
    localAlgorithm.input.set(covariance::data, localData);
    localAlgorithm.compute();

    /* Serialize partial results */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (rankId == mpi_root) {
        serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * comm_size]);
    }
    byte* nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);

    /* Gather partial results on root */
    MPI_Gather(nodeResults,
               perNodeArchLength,
               MPI_CHAR,
               serializedData.get(),
               perNodeArchLength,
               MPI_CHAR,
               mpi_root,
               MPI_COMM_WORLD);

    delete[] nodeResults;

    if (rankId == mpi_root) {
        /* Create an algorithm to compute a variance-covariance matrix on the master node */
        covariance::Distributed<step2Master> masterAlgorithm;

        for (size_t i = 0; i < comm_size; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i,
                                       perNodeArchLength);

            covariance::PartialResultPtr dataForStep2FromStep1 =
                covariance::PartialResultPtr(new covariance::PartialResult());

            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterAlgorithm.input.add(covariance::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute a dense variance-covariance matrix on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        covariance::ResultPtr result = masterAlgorithm.getResult();

        /* Print the results */
        printNumericTable(result->get(covariance::covariance), "Covariance matrix:");
        printNumericTable(result->get(covariance::mean), "Mean vector:");
    }

    MPI_Finalize();

    return 0;
}
