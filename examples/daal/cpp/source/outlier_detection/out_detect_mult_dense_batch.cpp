/* file: out_detect_mult_dense_batch.cpp */
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
!    C++ example of multivariate outlier detection
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-OUTLIER_DETECTION_MULTIVARIATE_DENSE_BATCH"></a>
 * \example out_detect_mult_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

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

    /* Create an algorithm to detect outliers using the default method */
    multivariate_outlier_detection::Batch<float, multivariate_outlier_detection::defaultDense>
        algorithm;

    algorithm.input.set(multivariate_outlier_detection::data, dataSource.getNumericTable());

    /* Compute outliers */
    algorithm.compute();

    /* Get the computed results */
    multivariate_outlier_detection::ResultPtr res = algorithm.getResult();

    printNumericTables(dataSource.getNumericTable().get(),
                       res->get(multivariate_outlier_detection::weights).get(),
                       "Input data",
                       "Weights",
                       "Outlier detection result (Default method)");

    return 0;
}
