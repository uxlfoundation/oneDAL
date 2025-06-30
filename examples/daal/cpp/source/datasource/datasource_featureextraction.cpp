/* file: datasource_featureextraction.cpp */
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
!    C++ example for using of data source feature extraction
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASOURCE_FEATUREEXTRACTION"></a>
 * \example datasource_featureextraction.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/kmeans_dense.csv";

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable);

    /* Create data source dictionary from loading of the first .csv file */
    dataSource.createDictionaryFromContext();

    /* Filter in 3 chosen columns from a .csv file */
    services::Collection<size_t> validList(3);
    validList[0] = 1;
    validList[1] = 2;
    validList[2] = 5;

    dataSource.getFeatureManager().addModifier(ColumnFilter().list(validList));

    /* Consider column with index 1 as categorical and convert it into 3 binary categorical features */
    dataSource.getFeatureManager().addModifier(OneHotEncoder(1, 3));

    /* Load data from .csv file */
    dataSource.loadDataBlock();

    /* Print result */
    NumericTablePtr table = dataSource.getNumericTable();
    printNumericTable(table, "Loaded data", 4, 20);

    return 0;
}
