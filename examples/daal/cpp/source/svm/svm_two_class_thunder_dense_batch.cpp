/* file: svm_two_class_thunder_dense_batch.cpp */
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
!    C++ example of two-class support vector machine (SVM) classification using
!    the Thunder method
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_THUNDER_DENSE_BATCH"></a>
 * \example svm_two_class_thunder_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/svm_two_class_train_dense_data.csv";
const std::string trainDatasetLabelFileName = "data/svm_two_class_train_dense_label.csv";
const std::string testDatasetFileName = "data/svm_two_class_test_dense_data.csv";
const std::string testDatasetLabelFileName = "data/svm_two_class_test_dense_label.csv";

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

void trainModel();
void testModel();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 4, &trainDatasetFileName, &trainDatasetLabelFileName, &testDatasetFileName, &testDatasetLabelFileName);

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

    /* Create an algorithm object to train the SVM model using the Thunder method */
    svm::training::Batch<float, svm::training::thunder> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainDataSource.getNumericTable());
    algorithm.input.set(classifier::training::labels, trainLabelSource.getNumericTable());

    /* Build the SVM model */
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

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testDataSource.getNumericTable());
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults() {
    FileDataSource<CSVFeatureManager> testLabelSource(testDatasetLabelFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testLabelSource.loadDataBlock();

    printNumericTables<int, float>(testLabelSource.getNumericTable(),
                                   predictionResult->get(classifier::prediction::prediction),
                                   "Ground truth\t",
                                   "Classification results",
                                   "SVM classification results (first 20 observations):",
                                   20);
}
