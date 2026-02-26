/* file: covariance_csr_distributed_mpi.cpp */
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
!    C++ sample of sparse variance-covariance matrix computation in the
!    distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-COVARIANCE_CSR_DISTRIBUTED"></a>
 * \example covariance_csr_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType; /* Algorithm floating-point type */

int rankId, comm_size;
#define mpi_root 0

const std::string datasetFileName = "data/covcormoments_csr.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    CSRNumericTablePtr fullData(createSparseTable<algorithmFPType>(datasetFileName));

    const size_t totalRows = fullData->getNumberOfRows();

    /* Split data according to MPI ranks */
    const size_t rowsPerRank = (totalRows + comm_size - 1) / comm_size;

    const size_t rowStart = rankId * rowsPerRank;
    const size_t rowEnd = std::min(rowStart + rowsPerRank, totalRows);

    CSRNumericTablePtr localTable = splitCSRBlock<algorithmFPType>(fullData, rowStart, rowEnd);

    /* Create an algorithm to compute a sparse variance-covariance matrix on local nodes */
    covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR> localAlgorithm;

    localAlgorithm.input.set(covariance::data, localTable);
    localAlgorithm.compute();

    /* Serialize partial results */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    byte* nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);

    if (rankId == mpi_root) {
        serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * comm_size]);
    }

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
        covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR> masterAlgorithm;

        for (int i = 0; i < comm_size; i++) {
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i,
                                       perNodeArchLength);

            covariance::PartialResultPtr dataForStep2FromStep1(new covariance::PartialResult());

            dataForStep2FromStep1->deserialize(dataArch);

            masterAlgorithm.input.add(covariance::partialResults, dataForStep2FromStep1);
        }

        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        covariance::ResultPtr result = masterAlgorithm.getResult();

        printNumericTable(result->get(covariance::covariance),
                          "Covariance matrix (upper left 10x10):",
                          10,
                          10);
        printNumericTable(result->get(covariance::mean), "Mean vector:", 1, 10);
    }

    MPI_Finalize();
    return 0;
}
