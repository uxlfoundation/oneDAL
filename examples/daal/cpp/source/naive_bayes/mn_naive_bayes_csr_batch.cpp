/* file: mn_naive_bayes_csr_batch.cpp */
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
!    C++ example of Naive Bayes classification in the batch processing mode.
!
!    The program trains the Naive Bayes model on a supplied training data set in
!    compressed sparse rows (CSR) format and then performs classification of
!    previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MULTINOMIAL_NAIVE_BAYES_CSR_BATCH"></a>
 * \example mn_naive_bayes_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::multinomial_naive_bayes;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/naivebayes_train_csr.csv";
const std::string trainGroundTruthFileName = "../data/batch/naivebayes_train_labels.csv";

const std::string testDatasetFileName = "../data/batch/naivebayes_test_csr.csv";
const std::string testGroundTruthFileName = "../data/batch/naivebayes_test_labels.csv";

const size_t nTrainObservations = 8000;
const size_t nTestObservations = 2000;
const size_t nClasses = 20;

training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

void trainModel();
void testModel();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainGroundTruthSource(trainGroundTruthFileName,
                                                             DataSource::doAllocateNumericTable,
                                                             DataSource::doDictionaryFromContext);

    /* Retrieve the data from input files */
    CSRNumericTablePtr trainData(createSparseTable<float>(trainDatasetFileName));
    trainGroundTruthSource.loadDataBlock(nTrainObservations);

    /* Create an algorithm object to train the Naive Bayes model */
    training::Batch<algorithmFPType, training::fastCSR> algorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruthSource.getNumericTable());

    /* Build the Naive Bayes model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    CSRNumericTablePtr testData(createSparseTable<float>(testDatasetFileName));

    /* Create an algorithm object to predict Naive Bayes values */
    prediction::Batch<algorithmFPType, prediction::fastCSR> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict Naive Bayes values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults() {
    FileDataSource<CSVFeatureManager> testGroundTruth(testGroundTruthFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testGroundTruth.loadDataBlock(nTestObservations);

    printNumericTables<int, int>(testGroundTruth.getNumericTable().get(),
                                 predictionResult->get(classifier::prediction::prediction).get(),
                                 "Ground truth",
                                 "Classification results",
                                 "NaiveBayes classification results (first 20 observations):",
                                 20);
}
