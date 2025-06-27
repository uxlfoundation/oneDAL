/* file: cos_dist_dense_batch.cpp */
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
!    C++ example of computing a cosine distance matrix
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COSINE_DISTANCE_BATCH"></a>
 * \example cos_dist_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/distance.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute a cosine distance matrix using the default method */
    cosine_distance::Batch<> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(cosine_distance::data, dataSource.getNumericTable());

    /* Compute a cosine distance matrix */
    algorithm.compute();

    /* Get the computed cosine distance matrix */
    cosine_distance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(cosine_distance::cosineDistance), "Cosine distance", 15);

    return 0;
}
