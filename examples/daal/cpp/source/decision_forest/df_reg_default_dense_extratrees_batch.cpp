/* file: df_reg_default_dense_extratrees_batch.cpp */
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
!    C++ example of decision forest regression in the batch processing mode
!    using the Extremely Randomized Trees algorithm.
!
!    The program trains the decision forest regression model on a training
!    datasetFileName and computes regression for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_REG_DEFAULT_DENSE_EXTRATREES_BATCH"></a>
 * \example df_reg_default_dense_extratrees_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::decision_forest::regression;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/df_regression_train.csv";
const std::string testDatasetFileName = "../data/batch/df_regression_test.csv";
const size_t categoricalFeaturesIndices[] = { 3 };
const size_t nFeatures = 13; /* Number of features in training and testing data sets */

/* Decision forest parameters */
const size_t nTrees = 100;

training::ResultPtr trainModel();
void testModel(const training::ResultPtr& res);
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    training::ResultPtr trainingResult = trainModel();
    testModel(trainingResult);

    return 0;
}

training::ResultPtr trainModel() {
    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData;
    NumericTablePtr trainDependentVariable;

    loadData(trainDatasetFileName, trainData, trainDependentVariable);

    /* Create an algorithm object to train the decision forest regression model with the default method */
    training::Batch<float, training::defaultDense> algorithm;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, trainData);
    algorithm.input.set(training::dependentVariable, trainDependentVariable);

    algorithm.parameter().nTrees = nTrees;
    algorithm.parameter().varImportance = daal::algorithms::decision_forest::training::MDI;
    /* Enable ExtraTrees regression algorithm with bootstrap=false and random splitter*/
    algorithm.parameter().splitter = daal::algorithms::decision_forest::training::random;
    algorithm.parameter().bootstrap = false;

    /* Build the decision forest regression model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    training::ResultPtr trainingResult = algorithm.getResult();
    printNumericTable(trainingResult->get(training::variableImportance),
                      "Variable importance results: ");
    return trainingResult;
}

void testModel(const training::ResultPtr& trainingResult) {
    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData;
    NumericTablePtr testGroundTruth;

    loadData(testDatasetFileName, testData, testGroundTruth);

    /* Create an algorithm object to predict values of decision forest regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testData);
    algorithm.input.set(prediction::model, trainingResult->get(training::model));

    /* Predict values of decision forest regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction),
                      "Decision forest prediction results (first 10 rows):",
                      10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData.reset(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    pDependentVar.reset(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(pData, pDependentVar));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());

    NumericTableDictionaryPtr pDictionary = pData->getDictionarySharedPtr();
    for (size_t i = 0,
                n = sizeof(categoricalFeaturesIndices) / sizeof(categoricalFeaturesIndices[0]);
         i < n;
         ++i)
        (*pDictionary)[categoricalFeaturesIndices[i]].featureType =
            data_feature_utils::DAAL_CATEGORICAL;
}
