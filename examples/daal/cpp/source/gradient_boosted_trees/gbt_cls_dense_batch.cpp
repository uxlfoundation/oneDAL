/* file: gbt_cls_dense_batch.cpp */
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
!    C++ example of gradient boosted trees classification in the batch processing mode.
!
!    The program trains the gradient boosted trees classification model on a training
!    datasetFileName and computes classification for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-GBT_CLS_DENSE_BATCH"></a>
 * \example gbt_cls_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::gbt::classification;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/df_classification_train_data.csv";
const std::string trainDatasetLabelFileName = "data/df_classification_train_label.csv";
const std::string testDatasetFileName = "data/df_classification_test_data.csv";
const std::string testDatasetLabelFileName = "data/df_classification_test_label.csv";

const size_t nFeatures = 3; /* Number of features in training and testing data sets */

/* Gradient boosted trees training parameters */
const size_t maxIterations = 40;
const size_t minObservationsInLeafNode = 8;

const size_t nClasses = 5; /* Number of classes */

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

    /* Create an algorithm object to train the gradient boosted trees classification model */
    training::Batch<> algorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainDataSource.getNumericTable());
    algorithm.input.set(classifier::training::labels, trainLabelSource.getNumericTable());

    algorithm.parameter().maxIterations = maxIterations;
    algorithm.parameter().featuresPerNode = nFeatures;
    algorithm.parameter().minObservationsInLeafNode = minObservationsInLeafNode;

    /* Build the gradient boosted trees classification model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    training::ResultPtr trainingResult = algorithm.getResult();
    return trainingResult;
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

    /* Create an algorithm object to predict values of gradient boosted trees classification */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testDataSource.getNumericTable());
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict values of gradient boosted trees classification */
    algorithm.compute();

    /* Retrieve the algorithm results */
    classifier::prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(classifier::prediction::prediction),
                      "Gragient boosted trees prediction results (first 10 rows):",
                      10);
    printNumericTable(testLabelSource.getNumericTable(), "Ground truth (first 10 rows):", 10);
}
