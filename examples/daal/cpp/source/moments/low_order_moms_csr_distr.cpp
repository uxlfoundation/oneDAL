/* file: low_order_moms_csr_distr.cpp */
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
!    C++ example of computing low order moments in the distributed processing
!    mode.
!
!    Input matrix is stored in the compressed sparse row (CSR) format with
!    one-based indexing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOW_ORDER_MOMENTS_CSR_DISTRIBUTED">
 * \example low_order_moms_csr_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileNames[] = { "../data/distributed/covcormoments_csr_1.csv",
                                         "../data/distributed/covcormoments_csr_2.csv",
                                         "../data/distributed/covcormoments_csr_3.csv",
                                         "../data/distributed/covcormoments_csr_4.csv" };

low_order_moments::PartialResultPtr partialResult[nBlocks];
low_order_moments::ResultPtr result;

void computestep1Local(size_t block);
void computeOnMasterNode();

void printResults(const low_order_moments::ResultPtr& res);

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    for (size_t block = 0; block < nBlocks; block++) {
        computestep1Local(block);
    }

    computeOnMasterNode();

    printResults(result);

    return 0;
}

void computestep1Local(size_t block) {
    CSRNumericTable* dataTable = createSparseTable<float>(datasetFileNames[block]);

    /* Create an algorithm to compute low order moments in the distributed processing mode using the default method */
    low_order_moments::Distributed<step1Local, algorithmFPType, low_order_moments::fastCSR>
        algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(low_order_moments::data, CSRNumericTablePtr(dataTable));

    /* Compute partial low order moments estimates on nodes */
    algorithm.compute();

    /* Get the computed partial estimates */
    partialResult[block] = algorithm.getPartialResult();
}

void computeOnMasterNode() {
    /* Create an algorithm to compute low order moments in the distributed processing mode using the default method */
    low_order_moments::Distributed<step2Master, algorithmFPType, low_order_moments::fastCSR>
        algorithm;

    /* Set input objects for the algorithm */
    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(low_order_moments::partialResults, partialResult[i]);
    }

    /* Compute a partial low order moments estimate on the master node from the partial estimates on local nodes */
    algorithm.compute();

    /* Finalize the result in the distributed processing mode */
    algorithm.finalizeCompute();

    /* Get the computed low order moments */
    result = algorithm.getResult();
}

void printResults(const low_order_moments::ResultPtr& res) {
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
