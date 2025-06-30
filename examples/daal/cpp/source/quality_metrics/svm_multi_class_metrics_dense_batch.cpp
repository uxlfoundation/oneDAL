/* file: svm_multi_class_metrics_dense_batch.cpp */
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
!    C++ example of multi-class support vector machine (SVM) quality metrics
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_MULTI_CLASS_QUALITY_METRIC_SET_BATCH"></a>
 * \example svm_multi_class_metrics_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::classifier::quality_metric;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/svm_multi_class_train_dense.csv";

const std::string testDatasetFileName = "../data/batch/svm_multi_class_test_dense.csv";

const size_t nFeatures = 20;
const size_t nClasses = 5;

services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());

/* Model object for the multi-class classifier algorithm */
multi_class_classifier::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

/* Parameters for the multi-class classifier kernel function */
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());

multi_class_classifier::quality_metric_set::ResultCollectionPtr qualityMetricSetResult;

NumericTablePtr predictedLabels;
NumericTablePtr groundTruthLabels;

void trainModel();
void testModel();
void testModelQuality();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    training->parameter.cacheSize = 100000000;
    training->parameter.kernel = kernel;
    prediction->parameter.kernel = kernel;

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
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    groundTruthLabels =
        NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, groundTruthLabels));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

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

void testModelQuality() {
    /* Retrieve predicted labels */
    predictedLabels = predictionResult->get(classifier::prediction::prediction);

    /* Create a quality metric set object to compute quality metrics of the multi-class classifier algorithm */
    multi_class_classifier::quality_metric_set::Batch qualityMetricSet(nClasses);

    multiclass_confusion_matrix::InputPtr input =
        qualityMetricSet.getInputDataCollection()->getInput(
            multi_class_classifier::quality_metric_set::confusionMatrix);

    input->set(multiclass_confusion_matrix::predictedLabels, predictedLabels);
    input->set(multiclass_confusion_matrix::groundTruthLabels, groundTruthLabels);

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
    multiclass_confusion_matrix::ResultPtr qualityMetricResult = qualityMetricSetResult->getResult(
        multi_class_classifier::quality_metric_set::confusionMatrix);
    printNumericTable(qualityMetricResult->get(multiclass_confusion_matrix::confusionMatrix),
                      "Confusion matrix:");

    BlockDescriptor<> block;
    NumericTablePtr qualityMetricsTable =
        qualityMetricResult->get(multiclass_confusion_matrix::multiClassMetrics);
    qualityMetricsTable->getBlockOfRows(0, 1, readOnly, block);
    float* qualityMetricsData = block.getBlockPtr();
    std::cout << "Average accuracy: "
              << qualityMetricsData[multiclass_confusion_matrix::averageAccuracy] << std::endl;
    std::cout << "Error rate:       " << qualityMetricsData[multiclass_confusion_matrix::errorRate]
              << std::endl;
    std::cout << "Micro precision:  "
              << qualityMetricsData[multiclass_confusion_matrix::microPrecision] << std::endl;
    std::cout << "Micro recall:     "
              << qualityMetricsData[multiclass_confusion_matrix::microRecall] << std::endl;
    std::cout << "Micro F-score:    "
              << qualityMetricsData[multiclass_confusion_matrix::microFscore] << std::endl;
    std::cout << "Macro precision:  "
              << qualityMetricsData[multiclass_confusion_matrix::macroPrecision] << std::endl;
    std::cout << "Macro recall:     "
              << qualityMetricsData[multiclass_confusion_matrix::macroRecall] << std::endl;
    std::cout << "Macro F-score:    "
              << qualityMetricsData[multiclass_confusion_matrix::macroFscore] << std::endl;
    qualityMetricsTable->releaseBlockOfRows(block);
}
