/* file: datastructures_merged.cpp */
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
!    Merged data structures example.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_MERGED">
 * \example datastructures_merged.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

int main() {
    std::cout << "Merged numeric table example" << std::endl << std::endl;

    const size_t nObservations = 5;
    const size_t nFeatures1 = 5;
    const size_t nFeatures2 = 6;
    const size_t firstReadRow = 3;
    const size_t nRead = 1;

    /*Example of using homogeneous numeric table*/
    float data1[nFeatures1 * nObservations] = {
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 2.0f, 2.1f, 2.2f,
        2.3f, 2.4f, 3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 4.0f, 4.1f, 4.2f, 4.3f, 4.4f,
    };
    float data2[nFeatures2 * nObservations] = {
        0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1,    1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2,    2.5f, 2.6f, 2.7f,
        2.8f, 2.9f, 3,    3.5f, 3.6f, 3.7f, 3.8f, 3.9f, 4,    4.5f, 4.6f, 4.7f, 4.8f, 4.9f, 5,
    };

    /* Create two homogen numeric tables from data arrays */
    NumericTablePtr dataTable1 = HomogenNumericTable<>::create(data1, nFeatures1, nObservations);
    checkPtr(dataTable1.get());
    NumericTablePtr dataTable2 = HomogenNumericTable<>::create(data2, nFeatures2, nObservations);
    checkPtr(dataTable2.get());

    /* Create merged numeric table consisting of two homogen numeric tables */
    MergedNumericTablePtr dataTable = MergedNumericTable::create();
    checkPtr(dataTable.get());
    dataTable->addNumericTable(dataTable1);
    dataTable->addNumericTable(dataTable2);

    BlockDescriptor<> block;

    /* Read one row from merged numeric table */
    dataTable->getBlockOfRows(firstReadRow, nRead, readWrite, block);
    printArray<float>(block.getBlockPtr(),
                      nFeatures1 + nFeatures2,
                      block.getNumberOfRows(),
                      "Print 1 row from merged numeric table as float:");

    /* Modify row of the merged numeric table */
    float* row = block.getBlockPtr();
    for (size_t i = 0; i < nFeatures1 + nFeatures2; i++)
        row[i] *= row[i];
    dataTable->releaseBlockOfRows(block);

    /* Read the same row from homogen numeric tables */
    dataTable1->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    printArray<float>(block.getBlockPtr(),
                      nFeatures1,
                      block.getNumberOfRows(),
                      "Print 1 row from first homogen numeric table as float:");
    dataTable1->releaseBlockOfRows(block);

    dataTable2->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    printArray<float>(block.getBlockPtr(),
                      nFeatures2,
                      block.getNumberOfRows(),
                      "Print 1 row from second homogen numeric table as float:");
    dataTable2->releaseBlockOfRows(block);

    return 0;
}
