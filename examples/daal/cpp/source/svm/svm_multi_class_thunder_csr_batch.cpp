/* file: svm_multi_class_thunder_csr_batch.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
!    the Thunder method
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_MULTI_CLASS_THUNDER_CSR_BATCH"></a>
 * \example svm_multi_class_thunder_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/svm_multi_class_train_csr.csv";
const std::string trainLabelsFileName = "../data/batch/svm_multi_class_train_labels.csv";

const std::string testDatasetFileName = "../data/batch/svm_multi_class_test_csr.csv";
const std::string testLabelsFileName = "../data/batch/svm_multi_class_test_labels.csv";

const size_t nClasses = 5;

services::SharedPtr<svm::training::Batch<float, svm::training::thunder> > training(
    new svm::training::Batch<float, svm::training::thunder>());
services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());

multi_class_classifier::training::ResultPtr trainingResult;
multi_class_classifier::prediction::ResultPtr predictionResult;
kernel_function::KernelIfacePtr kernel(
    new kernel_function::linear::Batch<float, kernel_function::linear::fastCSR>());
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &trainDatasetFileName,
                   &trainLabelsFileName,
                   &testDatasetFileName,
                   &testLabelsFileName);

    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

    trainModel();
    testModel();
    printResults();

    return 0;
}

void trainModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainLabelsDataSource(trainLabelsFileName,
                                                            DataSource::doAllocateNumericTable,
                                                            DataSource::doDictionaryFromContext);

    /* Create numeric table for training data */
    CSRNumericTablePtr trainData(createSparseTable<float>(trainDatasetFileName));

    /* Retrieve the data from the input file */
    trainLabelsDataSource.loadDataBlock();

    /* Create an algorithm object to train the multi-class SVM model */
    multi_class_classifier::training::Batch<> algorithm(nClasses);

    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainLabelsDataSource.getNumericTable());

    /* Build the multi-class SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Create Numeric Tables for testing data */
    NumericTablePtr testData(createSparseTable<float>(testDatasetFileName));

    /* Create an algorithm object to predict multi-class SVM values */
    multi_class_classifier::prediction::Batch<> algorithm(nClasses);

    algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

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
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testLabelsDataSource(testLabelsFileName,
                                                           DataSource::doAllocateNumericTable,
                                                           DataSource::doDictionaryFromContext);
    /* Retrieve the data from input file */
    testLabelsDataSource.loadDataBlock();
    testGroundTruth = testLabelsDataSource.getNumericTable();

    printNumericTables<int, int>(
        testGroundTruth,
        predictionResult->get(multi_class_classifier::prediction::prediction),
        "Ground truth",
        "Classification results",
        "Multi-class SVM classification sample program results (first 20 observations):",
        20);
}
