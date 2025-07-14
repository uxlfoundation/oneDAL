/* file: svm_multi_class_boser_dense_batch.cpp */
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
!    C++ example of multi-class support vector machine (SVM) classification using
!    the Boser method
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_MULTI_CLASS_BOSER_DENSE_BATCH"></a>
 * \example svm_multi_class_boser_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/svm_multi_class_train_dense.csv";
const std::string testDatasetFileName = "../data/batch/svm_multi_class_test_dense.csv";

const size_t nFeatures = 20;
const size_t nClasses = 5;

services::SharedPtr<svm::training::Batch<float, svm::training::boser> > training(
    new svm::training::Batch<float, svm::training::boser>());
services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());

multi_class_classifier::training::ResultPtr trainingResult;
multi_class_classifier::prediction::ResultPtr predictionResult;
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    trainModel();
    testModel();
    printResults();

    return 0;
}

void trainModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData =
        HomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    NumericTablePtr trainGroundTruth =
        HomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData = MergedNumericTable::create(trainData, trainGroundTruth);

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the multi-class SVM model */
    multi_class_classifier::training::Batch<> algorithm(nClasses);

    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Build the multi-class SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData =
        HomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    testGroundTruth = HomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData = MergedNumericTable::create(testData, testGroundTruth);

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict multi-class SVM values */
    multi_class_classifier::prediction::Batch<float, multi_class_classifier::prediction::voteBased>
        algorithm(nClasses);

    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;
    algorithm.parameter.resultsToEvaluate = multi_class_classifier::computeClassLabels |
                                            multi_class_classifier::computeDecisionFunction;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict multi-class SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults() {
    printNumericTables<int, int>(
        testGroundTruth,
        predictionResult->get(multi_class_classifier::prediction::prediction),
        "Ground truth",
        "Classification results",
        "Multi-class SVM classification sample program results (first 20 observations):",
        20);
    printNumericTable(
        predictionResult->get(multi_class_classifier::prediction::decisionFunction),
        "Multi-class SVM classification decision function results (first 20 observations):",
        20);
}
