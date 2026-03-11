/* file: df_cls_hist_dense_extratrees_batch.cpp */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
!    C++ example of decision forest classification in the batch processing mode
!    using the Extremely Randomized Trees algorithm.
!
!    The program trains the decision forest classification model on a training
!    datasetFileName and computes classification for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_CLS_HIST_DENSE_EXTRATREES_BATCH"></a>
 * \example df_cls_hist_dense_extratrees_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::decision_forest::classification;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/df_classification_train_data.csv";
const std::string trainDatasetLabelFileName = "data/df_classification_train_label.csv";
const std::string testDatasetFileName = "data/df_classification_test_data.csv";
const std::string testDatasetLabelFileName = "data/df_classification_test_label.csv";

const size_t nFeatures = 3; /* Number of features in training and testing data sets */

/* Decision forest parameters */
const size_t nTrees = 10;
const size_t minObservationsInLeafNode = 8;
const size_t minObservationsInSplitNode = 16;
const double minWeightFractionInLeafNode = 0.0; /* It must be in segment [0.0, 0.5] */
const double minImpurityDecreaseInSplitNode = 0.0; /* It must be greater than or equal to 0.0 */
const size_t maxBins = 256; /* Default value */
const size_t minBinSize = 5; /* Default value */

const size_t nClasses = 5; /* Number of classes */

training::ResultPtr trainModel();
void testModel(const training::ResultPtr& res);

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &trainDatasetFileName,
                   &trainDatasetLabelFileName,
                   &testDatasetFileName,
                   &testDatasetLabelFileName);

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

    /* Create an algorithm object to train the decision forest classification model */
    training::Batch<float, training::hist> algorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainDataSource.getNumericTable());
    algorithm.input.set(classifier::training::labels, trainLabelSource.getNumericTable());

    algorithm.parameter().nTrees = nTrees;
    algorithm.parameter().featuresPerNode = nFeatures;
    algorithm.parameter().minObservationsInLeafNode = minObservationsInLeafNode;
    algorithm.parameter().minObservationsInSplitNode = minObservationsInSplitNode;
    algorithm.parameter().minWeightFractionInLeafNode = minWeightFractionInLeafNode;
    algorithm.parameter().minImpurityDecreaseInSplitNode = minImpurityDecreaseInSplitNode;
    algorithm.parameter().varImportance = algorithms::decision_forest::training::MDI;
    algorithm.parameter().maxBins = maxBins;
    algorithm.parameter().minBinSize = minBinSize;
    /* Enable ExtraTrees classification algorithm with bootstrap=false and random splitter*/
    algorithm.parameter().splitter = algorithms::decision_forest::training::random;
    algorithm.parameter().bootstrap = false;

    /* Build the decision forest classification model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    training::ResultPtr trainingResult = algorithm.getResult();
    printNumericTable(trainingResult->get(training::variableImportance),
                      "Variable importance results: ");
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

    /* Create an algorithm object to predict values of decision forest classification */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testDataSource.getNumericTable());
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));
    algorithm.parameter().votingMethod = prediction::weighted;
    algorithm.parameter().resultsToEvaluate |= classifier::computeClassProbabilities;
    /* Predict values of decision forest classification */
    algorithm.compute();

    /* Retrieve the algorithm results */
    classifier::prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(classifier::prediction::prediction),
                      "Decision forest prediction results (first 10 rows):",
                      10);
    printNumericTable(predictionResult->get(classifier::prediction::probabilities),
                      "Decision forest probabilities results (first 10 rows):",
                      10);
    printNumericTable(testLabelSource.getNumericTable(), "Ground truth (first 10 rows):", 10);
}
