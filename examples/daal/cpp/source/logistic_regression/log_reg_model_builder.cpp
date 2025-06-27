/* file: log_reg_model_builder.cpp */
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

/**
 * <a name="DAAL-EXAMPLE-CPP-LOGISTIC_REGRESSION_MODEL_BUILDER"></a>
 * \example log_reg_model_builder.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::logistic_regression;

/* Input data set parameters */
const std::string trainedModelFileName = "../data/batch/logreg_trained_model.csv";
const std::string testDatasetFileName = "../data/batch/logreg_test.csv";

const size_t nFeatures = 6; /* Number of features in training and testing data sets */
const size_t nClasses = 5; /* Number of classes */

ModelPtr buildModel();
void testModel(ModelPtr &);

int main(int argc, char *argv[]) {
    checkArguments(argc, argv, 2, &trainedModelFileName, &testDatasetFileName);

    ModelPtr builtModel = buildModel();
    testModel(builtModel);

    return 0;
}

ModelPtr buildModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the beta data from a .csv file */
    FileDataSource<CSVFeatureManager> modelSource(trainedModelFileName,
                                                  DataSource::doAllocateNumericTable,
                                                  DataSource::doDictionaryFromContext);

    /* Create Numeric Table for beta coefficients */
    NumericTablePtr beta(new HomogenNumericTable<>(nFeatures + 1, 0, NumericTable::doNotAllocate));
    /* Get beta from trained model */
    modelSource.loadDataBlock(beta.get());

    /* Retrieve  pointer to the beginning of beta */
    BlockDescriptor<> blockResult;
    beta->getBlockOfRows(0, nClasses, readOnly, blockResult);
    /* Define the size of beta */
    size_t numberOfBetas = (beta->getNumberOfRows()) * (beta->getNumberOfColumns());

    /* Initialize iterators for beta array with itrecepts */
    float *first = blockResult.getBlockPtr();
    float *last = first + numberOfBetas;

    /* Create model builder with true intercept flag */
    ModelBuilder<> modelBuilder(nFeatures, nClasses);

    /* Set beta */
    modelBuilder.setBeta(first, last);
    beta->releaseBlockOfRows(blockResult);

    printNumericTable(modelBuilder.getModel()->getBeta(),
                      "Logistic Regression coefficients of built model:");

    return modelBuilder.getModel();
}

void testModel(ModelPtr &inputModel) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr testGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Load the data from the data file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict values of logistic regression */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(algorithms::classifier::prediction::data, testData);
    algorithm.input.set(algorithms::classifier::prediction::model, inputModel);

    /* Predict values of logistic regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    NumericTablePtr prediction =
        algorithm.getResult()->get(algorithms::classifier::prediction::prediction);
    printNumericTable(prediction, "Logistic Regression prediction results: (first 10 rows):", 10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}
