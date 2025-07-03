/* file: assoc_rules_apriori_batch.cpp */
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
!    C++ example of association rules mining
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-APRIORI_BATCH"></a>
 * \example assoc_rules_apriori_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string datasetFileName = "../data/batch/apriori.csv";

/* Apriori algorithm parameters */
const double minSupport = 0.001; /* Minimum support */
const double minConfidence = 0.7; /* Minimum confidence */

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to mine association rules using the Apriori method */
    association_rules::Batch<> algorithm;

    /* Set the input object for the algorithm */
    algorithm.input.set(association_rules::data, dataSource.getNumericTable());

    /* Set the Apriori algorithm parameters */
    algorithm.parameter.minSupport = minSupport;
    algorithm.parameter.minConfidence = minConfidence;

    /* Find large item sets and construct association rules */
    algorithm.compute();

    /* Get computed results of the Apriori algorithm */
    association_rules::ResultPtr res = algorithm.getResult();

    /* Print the large item sets */
    printAprioriItemsets(res->get(association_rules::largeItemsets),
                         res->get(association_rules::largeItemsetsSupport));

    /* Print the association rules */
    printAprioriRules(res->get(association_rules::antecedentItemsets),
                      res->get(association_rules::consequentItemsets),
                      res->get(association_rules::confidence));

    return 0;
}
