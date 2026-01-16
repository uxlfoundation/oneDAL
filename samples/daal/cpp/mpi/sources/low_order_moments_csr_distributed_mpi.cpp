/* file: low_order_moments_csr_distributed_mpi.cpp */
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
!    mode.
!
!    Input matrix is stored in the compressed sparse row (CSR) format.
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-LOW_ORDER_MOMENTS_CSR_DISTRIBUTED"></a>
 * \example low_order_moments_csr_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nBlocks = 4;

typedef float algorithmFPType; /* Algorithm floating-point type */

int rankId, comm_size;
#define mpi_root 0

const std::string datasetFileNames[] = { "data/covcormoments_csr_1.csv",
                                         "data/covcormoments_csr_2.csv",
                                         "data/covcormoments_csr_3.csv",
                                         "data/covcormoments_csr_4.csv" };

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    /* Retrieve the input data from a file */
    CSRNumericTable* dataTable = createSparseTable<float>(datasetFileNames[rankId]);

    /* Create an algorithm to compute low order moments on local nodes */
    low_order_moments::Distributed<step1Local, algorithmFPType, low_order_moments::fastCSR>
        localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(low_order_moments::data, CSRNumericTablePtr(dataTable));

    /* Compute low order moments */
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
        low_order_moments::Distributed<step2Master, algorithmFPType, low_order_moments::fastCSR>
            masterAlgorithm;

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
