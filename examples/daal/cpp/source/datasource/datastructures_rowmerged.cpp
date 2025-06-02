/* file: datastructures_rowmerged.cpp */
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
!    Row merged data structures example.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_ROWMERGED">
 * \example datastructures_rowmerged.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

int main() {
    std::cout << "Row merged numeric table example" << std::endl << std::endl;

    const size_t nObservations1 = 5;
    const size_t nObservations2 = 6;
    const size_t nFeatures = 5;
    const size_t featureIdx = 2;

    /*Example of using homogeneous numeric table*/
    float data1[nFeatures * nObservations1] = {
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 2.0f, 2.1f, 2.2f,
        2.3f, 2.4f, 3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 4.0f, 4.1f, 4.2f, 4.3f, 4.4f,
    };
    float data2[nFeatures * nObservations2] = {
        0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f,
        3.5f, 3.6f, 3.7f, 3.8f, 3.9f, 4.5f, 4.6f, 4.7f, 4.8f, 4.9f, 5.5f, 5.6f, 5.7f, 5.8f, 5.9f,
    };

    /* Create row merged numeric table consisting of two homogen numeric tables */

    NumericTablePtr table1 =
        HomogenNumericTable<>::create(DictionaryIface::equal, data1, nFeatures, nObservations1);
    checkPtr(table1.get());
    NumericTablePtr table2 =
        HomogenNumericTable<>::create(DictionaryIface::equal, data2, nFeatures, nObservations2);
    checkPtr(table2.get());

    RowMergedNumericTablePtr dataTable = RowMergedNumericTable::create();
    checkPtr(dataTable.get());
    dataTable->addNumericTable(table1);
    dataTable->addNumericTable(table2);

    BlockDescriptor<> block;

    /* Read one row from merged numeric table */
    dataTable->getBlockOfRows(0, nObservations1 + nObservations2, readWrite, block);
    printArray<float>(block.getBlockPtr(),
                      nFeatures,
                      block.getNumberOfRows(),
                      "Print rows from row merged numeric table as float:");

    /* Modify row of the merged numeric table */
    float* row = block.getBlockPtr();
    for (size_t i = 0; i < nObservations1 + nObservations2; i++)
        row[i * nFeatures + featureIdx] *= row[i * nFeatures + featureIdx];
    dataTable->releaseBlockOfRows(block);

    dataTable->getBlockOfRows(0, nObservations1 + nObservations2, readOnly, block);
    printArray<float>(block.getBlockPtr(),
                      nFeatures,
                      block.getNumberOfRows(),
                      "Print rows from row merged numeric table as float:");
    dataTable->releaseBlockOfRows(block);

    NumericTablePtr finalizedTable = data_management::convertToHomogen<float>(*dataTable);

    printNumericTable(finalizedTable.get(), "Row merged table converted to homogen numeric table");

    return 0;
}
