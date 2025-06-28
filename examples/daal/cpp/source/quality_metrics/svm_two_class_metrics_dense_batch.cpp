/* file: svm_two_class_metrics_dense_batch.cpp */
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
!    C++ example of two-class support vector machine (SVM) quality metrics
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_QUALITY_METRIC_SET_BATCH"></a>
 * \example svm_two_class_metrics_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::classifier::quality_metric;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/svm_two_class_train_dense.csv";

const std::string testDatasetFileName = "../data/batch/svm_two_class_test_dense.csv";

const size_t nFeatures = 20;

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());

/* Model object for the SVM algorithm */
svm::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
svm::quality_metric_set::ResultCollectionPtr qualityMetricSetResult;

NumericTablePtr predictedLabels;
NumericTablePtr groundTruthLabels;

void trainModel();
void testModel();
void testModelQuality();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();

    testModel();

    testModelQuality();

    printResults();

    return 0;
}

void trainModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to train the SVM model */
    svm::training::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;
    algorithm.parameter.cacheSize = 40000000;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);

    /* Build the SVM model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    groundTruthLabels =
        NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, groundTruthLabels));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void testModelQuality() {
    /* Retrieve predicted labels */
    predictedLabels = predictionResult->get(classifier::prediction::prediction);

    /* Create a quality metric set object to compute quality metrics of the SVM algorithm */
    svm::quality_metric_set::Batch qualityMetricSet;

    binary_confusion_matrix::InputPtr input = qualityMetricSet.getInputDataCollection()->getInput(
        svm::quality_metric_set::confusionMatrix);
    input->set(binary_confusion_matrix::predictedLabels, predictedLabels);
    input->set(binary_confusion_matrix::groundTruthLabels, groundTruthLabels);

    /* Compute quality metrics */
    qualityMetricSet.compute();

    /* Retrieve the quality metrics */
    qualityMetricSetResult = qualityMetricSet.getResultCollection();
}

void printResults() {
    /* Print the classification results */
    printNumericTables<int, float>(groundTruthLabels.get(),
                                   predictedLabels.get(),
                                   "Ground truth",
                                   "Classification results",
                                   "SVM classification results (first 20 observations):",
                                   20);

    /* Print the quality metrics */
    binary_confusion_matrix::ResultPtr qualityMetricResult =
        qualityMetricSetResult->getResult(svm::quality_metric_set::confusionMatrix);
    printNumericTable(qualityMetricResult->get(binary_confusion_matrix::confusionMatrix),
                      "Confusion matrix:");

    BlockDescriptor<> block;
    NumericTablePtr qualityMetricsTable =
        qualityMetricResult->get(binary_confusion_matrix::binaryMetrics);
    qualityMetricsTable->getBlockOfRows(0, 1, readOnly, block);
    float* qualityMetricsData = block.getBlockPtr();
    std::cout << "Accuracy:      " << qualityMetricsData[binary_confusion_matrix::accuracy]
              << std::endl;
    std::cout << "Precision:     " << qualityMetricsData[binary_confusion_matrix::precision]
              << std::endl;
    std::cout << "Recall:        " << qualityMetricsData[binary_confusion_matrix::recall]
              << std::endl;
    std::cout << "F-score:       " << qualityMetricsData[binary_confusion_matrix::fscore]
              << std::endl;
    std::cout << "Specificity:   " << qualityMetricsData[binary_confusion_matrix::specificity]
              << std::endl;
    std::cout << "AUC:           " << qualityMetricsData[binary_confusion_matrix::AUC] << std::endl;
    qualityMetricsTable->releaseBlockOfRows(block);
}
