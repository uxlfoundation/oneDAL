/* file: zscore_dense_batch.cpp */
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
!    C++ example of Z-score normalization algorithm.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ZSCORE_BATCH"></a>
 * \example zscore_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::normalization;

/* Input data set parameters */
const std::string datasetName = "../data/batch/normalization.csv";

int main() {
    /* Retrieve the input data */
    FileDataSource<CSVFeatureManager> dataSource(datasetName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();

    NumericTablePtr data = dataSource.getNumericTable();

    /* Create an algorithm */
    zscore::Batch<float, zscore::sumDense> algorithm;

    /* Set an input object for the algorithm */
    algorithm.input.set(zscore::data, data);

    /* Compute Z-score normalization function */
    algorithm.compute();

    /* Print the results of stage */
    zscore::ResultPtr res = algorithm.getResult();

    printNumericTable(data, "First 10 rows of the input data:", 10);
    printNumericTable(res->get(zscore::normalizedData),
                      "First 10 rows of the z-score normalization result:",
                      10);

    return 0;
}
