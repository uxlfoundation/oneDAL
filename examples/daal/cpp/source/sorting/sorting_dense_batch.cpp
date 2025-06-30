/* file: sorting_dense_batch.cpp */
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
!    C++ example of sorting the observations matrix
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SORTING_BATCH"></a>
 * \example sorting_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/sorting.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create algorithm objects to sort data using the default (radix) method */
    sorting::Batch<> algorithm;

    /* Print the input observations matrix */
    printNumericTable(dataSource.getNumericTable(), "Initial matrix of observations:");

    /* Set input objects for the algorithm */
    algorithm.input.set(sorting::data, dataSource.getNumericTable());

    /* Sort data observations */
    algorithm.compute();

    /* Get the sorting result */
    sorting::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(sorting::sortedData), "Sorted matrix of observations:");

    return 0;
}
