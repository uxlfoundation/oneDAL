/* file: serialization.cpp */
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
!    C++ example of numeric table serialization
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SERIALIZATION"></a>
 * \example serialization.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

/* Input data set parameters */
std::string datasetFileName = "../data/batch/serialization.csv";

void serializeNumericTable(NumericTablePtr dataTable, byte** buffer, size_t* length);
NumericTablePtr deserializeNumericTable(byte* buffer, size_t size);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Retrieve a numeric table */
    NumericTablePtr dataTable = dataSource.getNumericTable();

    /* Print the original data */
    printNumericTable(dataTable, "Data before serialization:");

    /* Serialize the numeric table into the memory buffer */
    byte* buffer;
    size_t length;
    serializeNumericTable(dataTable, &buffer, &length);

    /* Deserialize the numeric table from the memory buffer */
    NumericTablePtr restoredDataTable = deserializeNumericTable(buffer, length);

    /* Print the restored data */
    printNumericTable(restoredDataTable, "Data after deserialization:");

    delete[] buffer;
    return 0;
}

void serializeNumericTable(NumericTablePtr dataTable, byte** buffer, size_t* length) {
    /* Create a data archive to serialize the numeric table */
    InputDataArchive dataArch;

    /* Serialize the numeric table into the data archive */
    dataTable->serialize(dataArch);

    /* Get the length of the serialized data in bytes */
    *length = dataArch.getSizeOfArchive();

    /* Store the serialized data in an array */
    *buffer = new byte[*length];
    dataArch.copyArchiveToArray(*buffer, *length);
}

NumericTablePtr deserializeNumericTable(byte* buffer, size_t length) {
    /* Create a data archive to deserialize the numeric table */
    OutputDataArchive dataArch(buffer, length);

    /* Create a numeric table object */
    NumericTablePtr dataTable = NumericTablePtr(new HomogenNumericTable<>());

    /* Deserialize the numeric table from the data archive */
    dataTable->deserialize(dataArch);

    return dataTable;
}
