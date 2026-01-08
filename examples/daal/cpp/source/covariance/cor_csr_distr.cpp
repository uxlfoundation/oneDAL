/* file: cor_csr_distr.cpp */
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
!    C++ example of correlation matrix computation in the distributed
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CORRELATION_CSR_DISTRIBUTED">
 * \example cor_csr_distr.cpp
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

covariance::PartialResultPtr partialResult[nBlocks];
covariance::ResultPtr result;

void computeStep1Local(size_t block, const CSRNumericTablePtr& fullData) {
    const size_t totalRows = fullData->getNumberOfRows();
    const size_t rowsPerBlock = (totalRows + nBlocks - 1) / nBlocks;

    const size_t rowStart = block * rowsPerBlock;
    if (rowStart >= totalRows)
        return;

    const size_t rowEnd = std::min(rowStart + rowsPerBlock, totalRows);

    CSRNumericTablePtr localTable = splitCSRBlock<algorithmFPType>(fullData, rowStart, rowEnd);

    covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR> algorithm;

    algorithm.input.set(covariance::data, localTable);
    algorithm.compute();

    partialResult[block] = algorithm.getPartialResult();
}

void computeOnMasterNode() {
    covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR> algorithm;

    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(covariance::partialResults, partialResult[i]);
    }

    algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

    algorithm.compute();
    algorithm.finalizeCompute();

    result = algorithm.getResult();
}

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    CSRNumericTablePtr fullData(createSparseTable<algorithmFPType>(datasetFileName));

    for (size_t i = 0; i < nBlocks; i++) {
        computeStep1Local(i, fullData);
    }

    computeOnMasterNode();

    printNumericTable(result->get(covariance::correlation),
                      "Correlation matrix (upper left 10x10):",
                      10,
                      10);

    printNumericTable(result->get(covariance::mean), "Mean vector:", 1, 10);

    return 0;
}
