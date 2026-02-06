/* file: low_order_moments_dense_distributed_mpi.cpp */
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
!    C++ sample of computing low order moments in the distributed processing
!    mode
!
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-LOW_ORDER_MOMENTS_DENSE_DISTRIBUTED"></a>
 * \example low_order_moments_dense_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nBlocks = 4;

int rankId, comm_size;
#define mpi_root 0
const std::string datasetFileName = "data/covcormoments_dense.csv";

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    checkArguments(argc, argv, 1, &datasetFileName);

    /* 1. Count total rows (only root), broadcast to all */
    size_t totalRows = 0;
    if (rankId == mpi_root) {
        totalRows = countRowsCSV(datasetFileName);
    }
    MPI_Bcast(&totalRows, 1, MPI_UNSIGNED_LONG, mpi_root, MPI_COMM_WORLD);

    /* 2. Compute block for each process */
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

    /* 4. Compute local step */
    low_order_moments::Distributed<step1Local> localAlgorithm;
    localAlgorithm.input.set(low_order_moments::data, localData);
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (rankId == mpi_root) {
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
               mpi_root,
               MPI_COMM_WORLD);

    delete[] nodeResults;

    if (rankId == mpi_root) {
        /* Create an algorithm to compute low order moments on the master node */
        low_order_moments::Distributed<step2Master> masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i,
                                       perNodeArchLength);

            low_order_moments::PartialResultPtr dataForStep2FromStep1 =
                low_order_moments::PartialResultPtr(new low_order_moments::PartialResult());

            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm */
            masterAlgorithm.input.add(low_order_moments::partialResults, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute low order moments on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        low_order_moments::ResultPtr res = masterAlgorithm.getResult();

        /* Print the results */
        printNumericTable(res->get(low_order_moments::minimum), "Minimum:");
        printNumericTable(res->get(low_order_moments::maximum), "Maximum:");
        printNumericTable(res->get(low_order_moments::sum), "Sum:");
        printNumericTable(res->get(low_order_moments::sumSquares), "Sum of squares:");
        printNumericTable(res->get(low_order_moments::sumSquaresCentered),
                          "Sum of squared difference from the means:");
        printNumericTable(res->get(low_order_moments::mean), "Mean:");
        printNumericTable(res->get(low_order_moments::secondOrderRawMoment),
                          "Second order raw moment:");
        printNumericTable(res->get(low_order_moments::variance), "Variance:");
        printNumericTable(res->get(low_order_moments::standardDeviation), "Standard deviation:");
        printNumericTable(res->get(low_order_moments::variation), "Variation:");
    }

    MPI_Finalize();

    return 0;
}
