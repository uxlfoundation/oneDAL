/* file: low_order_moms_dense_batch.cpp */
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
!    C++ example of computing low order moments in the batch processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOW_ORDER_MOMENTS_DENSE_BATCH">
 * \example low_order_moms_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/covcormoments_dense.csv";

void printResults(const low_order_moments::ResultPtr& res);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute low order moments using the default method */
    low_order_moments::Batch<> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(low_order_moments::data, dataSource.getNumericTable());

    /* Compute low order moments */
    algorithm.compute();

    /* Get the computed low order moments */
    low_order_moments::ResultPtr res = algorithm.getResult();

    printResults(res);

    return 0;
}

void printResults(const low_order_moments::ResultPtr& res) {
    printNumericTable(res->get(low_order_moments::minimum), "Minimum:");
    printNumericTable(res->get(low_order_moments::maximum), "Maximum:");
    printNumericTable(res->get(low_order_moments::sum), "Sum:");
    printNumericTable(res->get(low_order_moments::sumSquares), "Sum of squares:");
    printNumericTable(res->get(low_order_moments::sumSquaresCentered),
                      "Sum of squared difference from the means:");
    printNumericTable(res->get(low_order_moments::mean), "Mean:");
    printNumericTable(res->get(low_order_moments::secondOrderRawMoment),
                      "Second order raw moment:");
    printNumericTable(res->get(low_order_moments::variance), "Variance:");
    printNumericTable(res->get(low_order_moments::standardDeviation), "Standard deviation:");
    printNumericTable(res->get(low_order_moments::variation), "Variation:");
}
