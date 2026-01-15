/* file: gbt_reg_dense_batch.cpp */
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
!    C++ example of gradient boosted trees regression in the batch processing mode.
!
!    The program trains the gradient boosted trees regression model on a training
!    datasetFileName and computes regression for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-GBT_REG_DENSE_BATCH"></a>
 * \example gbt_reg_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::gbt::regression;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/df_regression_train_data.csv";
const std::string trainDatasetLabelFileName = "data/df_regression_train_label.csv";
const std::string testDatasetFileName = "data/df_regression_test_data.csv";
const std::string testDatasetLabelFileName = "data/df_regression_test_label.csv";

/* Gradient boosted trees training parameters */
const size_t maxIterations = 40;

training::ResultPtr trainModel();
void testModel(const training::ResultPtr& res);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 4, &trainDatasetFileName, &trainDatasetLabelFileName, &testDatasetFileName, &testDatasetLabelFileName);

    training::ResultPtr trainingResult = trainModel();
    testModel(trainingResult);

    return 0;
}

training::ResultPtr trainModel() {
    /* Create Numeric Tables for training data and dependent variables */
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
     * from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    FileDataSource<CSVFeatureManager> trainLabelSource(trainDatasetLabelFileName,
                                                       DataSource::doAllocateNumericTable,
                                                       DataSource::doDictionaryFromContext);
    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock();
    trainLabelSource.loadDataBlock();

    /* Create an algorithm object to train the gradient boosted trees regression model with the default method */
    training::Batch<> algorithm;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, trainDataSource.getNumericTable());
    algorithm.input.set(training::dependentVariable, trainLabelSource.getNumericTable());

    algorithm.parameter().maxIterations = maxIterations;

    /* Build the gradient boosted trees regression model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    return algorithm.getResult();
}

void testModel(const training::ResultPtr& trainingResult) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
     * a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    testDataSource.loadDataBlock();
    FileDataSource<CSVFeatureManager> testLabelSource(testDatasetLabelFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testLabelSource.loadDataBlock();

    /* Create an algorithm object to predict values of gradient boosted trees regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testDataSource.getNumericTable());
    algorithm.input.set(prediction::model, trainingResult->get(training::model));

    /* Predict values of gradient boosted trees regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction),
                      "Gragient boosted trees prediction results (first 10 rows):",
                      10);
    printNumericTable(testLabelSource.getNumericTable(), "Ground truth (first 10 rows):", 10);
}
