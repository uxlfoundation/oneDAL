/* file: datastructures_homogen.cpp */
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
!    C++ example of using homogeneous data structures
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_HOMOGEN">
 * \example datastructures_homogen.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

int main() {
    std::cout << "Homogeneous numeric table example" << std::endl << std::endl;

    const size_t nObservations = 10;
    const size_t nFeatures = 11;
    const size_t firstReadRow = 0;
    const size_t nRead = 3;
    size_t readFeatureIdx;

    /*Example of using a homogeneous numeric table*/
    float data[nFeatures * nObservations] = {
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1,    1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2,    2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f,
        2.6f, 2.7f, 2.8f, 2.9f, 3,    3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f, 3.8f,
        3.9f, 4,    4.0f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f, 4.7f, 4.8f, 4.9f, 5,    5.0f,
        5.1f, 5.2f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f, 5.9f, 1,    6.0f, 6.1f, 6.2f, 6.3f,
        6.4f, 6.5f, 6.6f, 6.7f, 6.8f, 6.9f, 2,    7.0f, 7.1f, 7.2f, 7.3f, 7.4f, 7.5f, 7.6f,
        7.7f, 7.8f, 7.9f, 3,    8.0f, 8.1f, 8.2f, 8.3f, 8.4f, 8.5f, 8.6f, 8.7f, 8.8f, 8.9f,
        4,    9.0f, 9.1f, 9.2f, 9.3f, 9.4f, 9.5f, 9.6f, 9.7f, 9.8f, 9.9f, 5
    };

    services::SharedPtr<HomogenNumericTable<> > dataTable =
        HomogenNumericTable<>::create(data, nFeatures, nObservations);
    checkPtr(dataTable.get());

    BlockDescriptor<> block;

    /* Read a block of rows */
    dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl;
    printArray<float>(block.getBlockPtr(),
                      nFeatures,
                      block.getNumberOfRows(),
                      "Print 3 rows from homogeneous data array as float:");
    dataTable->releaseBlockOfRows(block);

    /* Read a feature(column) and write into it */
    readFeatureIdx = 2;
    dataTable->getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, block);
    printArray<float>(block.getBlockPtr(),
                      1,
                      block.getNumberOfRows(),
                      "Print the third feature of homogeneous data:");
    dataTable->releaseBlockOfColumnValues(block);

    /* Get a pointer to the inner array for HomogenNumericTable. This pointer is a pointer to the array data */
    data[0] = 999;
    float* dataFromNumericTable = dataTable->getArray();
    printArray<float>(dataFromNumericTable, nFeatures, nObservations, "Data from getArray:");

    const size_t nNewVectors = 3;
    float newData[nFeatures * nNewVectors] = {
        1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f,
        3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f, 3.0f,
        5.0f, 6.0f, 5.0f, 6.0f, 5.0f, 6.0f, 5.0f, 6.0f, 5.0f, 6.0f, 5.0f,
    };

    /* Set new data to HomogenNumericTable. It mush have the same type as the numeric table. */
    dataTable->setArray(newData, nNewVectors);

    /* Ensure the data has changed */
    readFeatureIdx = 1;
    dataTable->getBlockOfColumnValues(readFeatureIdx, firstReadRow, nNewVectors, readOnly, block);
    printArray<float>(block.getBlockPtr(),
                      1,
                      block.getNumberOfRows(),
                      "Print the second feature of new data:");
    dataTable->releaseBlockOfColumnValues(block);

    return 0;
}
