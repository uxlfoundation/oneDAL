/* file: datastructures_csr.cpp */
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
!    Compressed sparse rows (CSR) data structures example.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_CSR">
 * \example datastructures_csr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

int main() {
    std::cout << "Compressed spares rows (CSR) numeric table example" << std::endl << std::endl;

    const size_t nObservations = 5;
    const size_t nFeatures = 5;
    const size_t firstReadRow = 1;
    const size_t nRead = 3;

    /* Example of using CSR numeric table */
    float values[] = { 1, -1, -3, -2, 5, 4, 6, 4, -4, 2, 7, 8, -5 };
    size_t colIndices[] = { 1, 2, 4, 1, 2, 3, 4, 5, 1, 3, 4, 2, 5 };
    size_t rowOffsets[] = { 1, 4, 6, 9, 12, 14 };

    CSRNumericTablePtr dataTable =
        CSRNumericTable::create(values, colIndices, rowOffsets, nFeatures, nObservations);
    checkPtr(dataTable.get());

    /* Read block of rows in dense format */
    BlockDescriptor<> block;
    dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl << std::endl;
    printArray<float>(block.getBlockPtr(),
                      nFeatures,
                      block.getNumberOfRows(),
                      "Print 3 rows from CSR data array as dense float array:");
    dataTable->releaseBlockOfRows(block);

    /* Read block of rows in CSR format and write into it */
    CSRBlockDescriptor<> csrBlock;
    dataTable->getSparseBlock(firstReadRow, nRead, readWrite, csrBlock);
    float* valuesBlock = csrBlock.getBlockValuesPtr();
    size_t nValuesInBlock = csrBlock.getDataSize();
    printArray<float>(valuesBlock, nValuesInBlock, 1, "Values in 3 rows from CSR data array:");
    printArray<size_t>(csrBlock.getBlockColumnIndicesPtr(),
                       nValuesInBlock,
                       1,
                       "Columns indices in 3 rows from CSR data array:");
    printArray<size_t>(csrBlock.getBlockRowIndicesPtr(),
                       nRead + 1,
                       1,
                       "Rows offsets in 3 rows from CSR data array:");
    for (size_t i = 0; i < nValuesInBlock; i++) {
        valuesBlock[i] = -(1.0f + i);
    }
    dataTable->releaseSparseBlock(csrBlock);

    /* Read block of rows in dense format */
    dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl << std::endl;
    printArray<float>(block.getBlockPtr(),
                      nFeatures,
                      block.getNumberOfRows(),
                      "Print 3 rows from CSR data array as dense float array:");
    dataTable->releaseBlockOfRows(block);

    return 0;
}
