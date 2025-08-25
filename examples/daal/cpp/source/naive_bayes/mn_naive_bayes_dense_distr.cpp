/* file: mn_naive_bayes_dense_distr.cpp */
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
!    C++ example of Naive Bayes classification in the distributed processing
!    mode.
!
!    The program trains the Naive Bayes model on a supplied training data set in
!    dense format and then performs classification of previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MULTINOMIAL_NAIVE_BAYES_DENSE_DISTRIBUTED"></a>
 * \example mn_naive_bayes_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::multinomial_naive_bayes;

/* Input data set parameters */
const std::string trainDatasetFileNames[4] = { "../data/distributed/naivebayes_train_dense_1.csv",
                                               "../data/distributed/naivebayes_train_dense_2.csv",
                                               "../data/distributed/naivebayes_train_dense_3.csv",
                                               "../data/distributed/naivebayes_train_dense_4.csv" };

const std::string testDatasetFileName = "../data/distributed/naivebayes_test_dense.csv";

const size_t nFeatures = 20;
const size_t nClasses = 20;
const size_t nBlocks = 4;

void trainModel();
void testModel();
void printResults();

training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   5,
                   &trainDatasetFileNames[0],
                   &trainDatasetFileNames[1],
                   &trainDatasetFileNames[2],
                   &trainDatasetFileNames[3],
                   &testDatasetFileName);

    trainModel();
    testModel();

    printResults();

    return 0;
}

void trainModel() {
    training::Distributed<step2Master> masterAlgorithm(nClasses);

    for (size_t i = 0; i < nBlocks; i++) {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileNames[i],
                                                          DataSource::notAllocateNumericTable,
                                                          DataSource::doDictionaryFromContext);

        /* Create Numeric Tables for training data and labels */
        NumericTablePtr trainData(
            new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
        NumericTablePtr trainGroundTruth(
            new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
        NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

        /* Retrieve the data from the input file */
        trainDataSource.loadDataBlock(mergedData.get());

        /* Create an algorithm object to train the Naive Bayes model on the local-node data */
        training::Distributed<step1Local> localAlgorithm(nClasses);

        /* Pass a training data set and dependent values to the algorithm */
        localAlgorithm.input.set(classifier::training::data, trainData);
        localAlgorithm.input.set(classifier::training::labels, trainGroundTruth);

        /* Build the Naive Bayes model on the local node */
        localAlgorithm.compute();

        /* Set the local Naive Bayes model as input for the master-node algorithm */
        masterAlgorithm.input.add(training::partialModels, localAlgorithm.getPartialResult());
    }

    /* Merge and finalize the Naive Bayes model on the master node */
    masterAlgorithm.compute();
    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = masterAlgorithm.getResult();
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict Naive Bayes values */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, NumericTablePtr(testData));
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict Naive Bayes values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults() {
    printNumericTables<int, int>(testGroundTruth,
                                 predictionResult->get(classifier::prediction::prediction),
                                 "Ground truth",
                                 "Classification results",
                                 "NaiveBayes classification results (first 20 observations):",
                                 20);
}
