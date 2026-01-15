/* file: stump_cls_gini_dense_batch.cpp */
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
!    C++ example of stump classification.
!
!    The program trains the stump model on a supplied training datasetFileName
!    and then performs classification of previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-STUMP_CLS_GINI_DENSE_BATCH"></a>
 * \example stump_cls_gini_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::stump::classification;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/stump_train_data.csv";
const std::string trainDatasetLabelFileName = "data/stump_train_labels.csv";
const std::string testDatasetFileName = "data/stump_test_data.csv";
const std::string testDatasetLabelFileName = "data/stump_test_labels.csv";

training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

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

    printResults();

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
    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock();
    trainLabelSource.loadDataBlock();

    /* Create an algorithm object to train the stump model */
    training::Batch<> algorithm;
    algorithm.parameter().splitCriterion = decision_tree::classification::gini;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainDataSource.getNumericTable());
    algorithm.input.set(classifier::training::labels, trainLabelSource.getNumericTable());

    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
     * a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    testDataSource.loadDataBlock();
    /* Create an algorithm object to predict values */
    prediction::Batch<> algorithm;
    algorithm.parameter().resultsToEvaluate =
        classifier::computeClassLabels | classifier::computeClassProbabilities;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testDataSource.getNumericTable());
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults() {
    FileDataSource<CSVFeatureManager> testLabelSource(testDatasetLabelFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testLabelSource.loadDataBlock();
    printNumericTables<int, int>(testLabelSource.getNumericTable(),
                                 predictionResult->get(classifier::prediction::prediction),
                                 "Ground truth",
                                 "Classification results",
                                 "Stump classification results (first 20 observations):",
                                 20);

    printNumericTables<int, float>(testLabelSource.getNumericTable(),
                                   predictionResult->get(classifier::prediction::probabilities),
                                   "Ground truth",
                                   "Classification results",
                                   "Stump classification results (first 20 observations):",
                                   20);
}
