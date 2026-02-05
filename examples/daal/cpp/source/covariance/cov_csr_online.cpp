/* file: cov_csr_online.cpp */
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
!    C++ example of variance-covariance matrix computation in the online
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_CSR_ONLINE"></a>
 * \example cov_csr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileName = "data/covcormoments_csr.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    // Load full CSR (one-based)
    CSRNumericTablePtr fullData(createSparseTable<algorithmFPType>(datasetFileName));

    const size_t totalRows = fullData->getNumberOfRows();
    const size_t rowsPerBlock = (totalRows + nBlocks - 1) / nBlocks;

    /* Create an algorithm to compute a variance-covariance matrix in the online processing mode using the default method */
    covariance::Online<algorithmFPType, covariance::fastCSR> algorithm;

    for (size_t block = 0; block < nBlocks; ++block) {
        const size_t rowStart = block * rowsPerBlock;
        if (rowStart >= totalRows)
            break;

        const size_t rowEnd = std::min(rowStart + rowsPerBlock, totalRows);

        // split CSR exactly like in distributed
        CSRNumericTablePtr localTable = splitCSRBlock<algorithmFPType>(fullData, rowStart, rowEnd);

        /* Set input objects for the algorithm */
        algorithm.input.set(covariance::data, localTable);

        /* Compute partial estimates */
        algorithm.compute();
    }

    /* Finalize the result in the online processing mode */
    algorithm.finalizeCompute();

    /* Get the computed variance-covariance matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::covariance),
                      "Covariance matrix (upper left square 10*10) :",
                      10,
                      10);
    printNumericTable(res->get(covariance::mean), "Mean vector:", 1, 10);

    return 0;
}
