/* file: datastructures_aos.cpp */
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
!    C++ example of using an array of structures (AOS)
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_AOS">
 * @example datastructures_aos.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

struct PointType {
    float x;
    float y;
    int categ;
    double value;
};

int main() {
    std::cout << "Array of structures (AOS) numeric table example" << std::endl << std::endl;

    const size_t nObservations = 5;
    const size_t nFeatures = 4;
    PointType points[nObservations] = { { 0.5f, -1.3f, 1, 100.1 },
                                        { 2.5f, -3.3f, 2, 200.2 },
                                        { 4.5f, -5.3f, 2, 350.3 },
                                        { 6.5f, -7.3f, 0, 470.4 },
                                        { 8.5f, -9.3f, 1, 270.5 } };

    /* Construct AOS numericTable for a data array with nFeatures fields and nObservations elements*/
    AOSNumericTablePtr dataTable = AOSNumericTable::create(points, nFeatures, nObservations);
    checkPtr(dataTable.get());

    /* Add data to the numeric table */
    dataTable->setFeature<float>(0,
                                 DAAL_STRUCT_MEMBER_OFFSET(PointType, x),
                                 data_feature_utils::DAAL_CONTINUOUS);
    dataTable->setFeature<float>(1,
                                 DAAL_STRUCT_MEMBER_OFFSET(PointType, y),
                                 data_feature_utils::DAAL_CONTINUOUS);
    dataTable->setFeature<int>(2,
                               DAAL_STRUCT_MEMBER_OFFSET(PointType, categ),
                               data_feature_utils::DAAL_CATEGORICAL,
                               4);
    dataTable->setFeature<double>(3,
                                  DAAL_STRUCT_MEMBER_OFFSET(PointType, value),
                                  data_feature_utils::DAAL_CONTINUOUS);

    /* Read a block of rows */
    const size_t firstReadRow = 0;

    BlockDescriptor<double> doubleBlock;
    dataTable->getBlockOfRows(firstReadRow, nObservations, readOnly, doubleBlock);
    printArray<double>(doubleBlock.getBlockPtr(),
                       nFeatures,
                       doubleBlock.getNumberOfRows(),
                       "Print AOS data structures as double:");
    dataTable->releaseBlockOfRows(doubleBlock);

    /* Read a feature (column) */
    size_t readFeatureIdx = 2;

    BlockDescriptor<int> intBlock;
    dataTable->getBlockOfColumnValues(readFeatureIdx,
                                      firstReadRow,
                                      nObservations,
                                      readOnly,
                                      intBlock);
    printArray<int>(intBlock.getBlockPtr(),
                    1,
                    intBlock.getNumberOfRows(),
                    "Print the third feature of AOS:");
    dataTable->releaseBlockOfColumnValues(intBlock);

    return 0;
}
