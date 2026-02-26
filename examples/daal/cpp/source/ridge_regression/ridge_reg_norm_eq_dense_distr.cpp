/* file: ridge_reg_norm_eq_dense_distr.cpp */
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
!    C++ example of ridge regression in the distributed processing mode.
!
!    The program trains the multiple ridge regression model on a training
!    datasetFileName with the normal equations method and computes regression
!    for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-RIDGE_REGRESSION_NORM_EQ_DISTRIBUTED"></a>
 * \example ridge_reg_norm_eq_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::ridge_regression;

const std::string trainDatasetFileName = "data/linear_regression_train_data.csv";
const std::string trainDatasetLabelFileName = "data/linear_regression_train_responses.csv";
const std::string testDatasetFileName = "data/linear_regression_test_data.csv";
const std::string testDatasetLabelFileName = "data/linear_regression_test_responses.csv";

const size_t nBlocks = 4;

void trainModel();
void testModel();

training::ResultPtr trainingResult;
prediction::ResultPtr predictionResult;

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &trainDatasetFileName,
                   &trainDatasetLabelFileName,
                   &testDatasetFileName,
                   &testDatasetLabelFileName);

    trainModel();
    testModel();

    return 0;
}

void trainModel() {
    /* Create Numeric Tables for training data and dependent variables */
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
     * from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    FileDataSource<CSVFeatureManager> trainLabelSource(trainDatasetLabelFileName,
                                                       DataSource::doAllocateNumericTable,
                                                       DataSource::doDictionaryFromContext);
    /* Create an algorithm object to build the final multiple linear regression model on the master node */
    training::Distributed<step2Master> masterAlgorithm;
    size_t totalRows = countRowsCSV(trainDatasetFileName);
    size_t blockSize = (totalRows + nBlocks - 1) / nBlocks;

    size_t remainingRows = totalRows;

    for (size_t i = 0; i < nBlocks; i++) {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        size_t rowsToRead = std::min(blockSize, remainingRows);
        size_t nLoaded = trainDataSource.loadDataBlock(rowsToRead);
        trainLabelSource.loadDataBlock(rowsToRead);
        remainingRows -= nLoaded;
        /* Create an algorithm object to train the ridge regression model based on the local-node data */
        training::Distributed<step1Local> localAlgorithm;

        /* Pass a training data set and dependent values to the algorithm */
        localAlgorithm.input.set(training::data, trainDataSource.getNumericTable());
        localAlgorithm.input.set(training::dependentVariables, trainLabelSource.getNumericTable());

        /* Train the ridge regression model on the local-node data */
        localAlgorithm.compute();

        /* Set the local ridge regression model as input for the master-node algorithm */
        masterAlgorithm.input.add(training::partialModels, localAlgorithm.getPartialResult());
    }

    /* Merge and finalize the ridge regression model on the master node */
    masterAlgorithm.compute();

    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = masterAlgorithm.getResult();
    printNumericTable(trainingResult->get(training::model)->getBeta(),
                      "Ridge Regression coefficients:");
}

void testModel() {
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
    /* Create an algorithm object to predict values of ridge regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testDataSource.getNumericTable());
    algorithm.input.set(prediction::model, trainingResult->get(training::model));

    /* Predict values of ridge regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction),
                      "Ridge Regression prediction results: (first 10 rows):",
                      10);
    printNumericTable(testLabelSource.getNumericTable(), "Ground truth (first 10 rows):", 10);
}
