/* file: out_detect_bacon_dense_batch.cpp */
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
!    C++ example of multivariate outlier detection using the Bacon method
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-OUT_DETECT_BACON_DENSE_BATCH"></a>
 * \example out_detect_bacon_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef double algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/outlierdetection.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to detect outliers using the BACON method */
    bacon_outlier_detection::Batch<algorithmFPType, bacon_outlier_detection::defaultDense>
        algorithm;

    algorithm.input.set(bacon_outlier_detection::data, dataSource.getNumericTable());

    /* Compute outliers */
    algorithm.compute();

    /* Get the computed results */
    bacon_outlier_detection::ResultPtr res = algorithm.getResult();

    printNumericTables(dataSource.getNumericTable().get(),
                       res->get(bacon_outlier_detection::weights).get(),
                       "Input data",
                       "Weights",
                       "Outlier detection result (Bacon method)");

    return 0;
}
