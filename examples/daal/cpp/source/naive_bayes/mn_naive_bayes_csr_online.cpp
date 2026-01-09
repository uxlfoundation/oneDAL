/* file: mn_naive_bayes_csr_online.cpp */
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
!    C++ example of Naive Bayes classification in the online processing mode.
!
!    The program trains the Naive Bayes model on a supplied training data set in
!    compressed sparse rows (CSR)__format and then performs classification of
!    previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MULTINOMIAL_NAIVE_BAYES_CSR_ONLINE"></a>
 * \example mn_naive_bayes_csr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::multinomial_naive_bayes;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const std::string trainDatasetFileName = "data/naivebayes_train_csr.csv";
const std::string trainGroundTruthFileName = "data/naivebayes_train_labels.csv";

const std::string testDatasetFileName = "data/naivebayes_test_csr.csv";
const std::string testGroundTruthFileName = "data/naivebayes_test_labels.csv";

const size_t nTestObservations = 2000;
const size_t nClasses = 20;
const size_t nBlocks = 4;

training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
CSRNumericTablePtr trainData[nBlocks];
CSRNumericTablePtr testData;

void trainModel();
void testModel();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &trainDatasetFileName);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void trainModel() {
    /* Create an algorithm object to train the Naive Bayes model */
    training::Online<algorithmFPType, training::fastCSR> algorithm(nClasses);
    CSRNumericTablePtr fullData(createSparseTable<float>(trainDatasetFileName));
    FileDataSource<CSVFeatureManager> dataSource(trainGroundTruthFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
    const size_t totalRows = fullData->getNumberOfRows();
    const size_t rowsPerBlock = (totalRows + nBlocks - 1) / nBlocks;
    size_t remainingRows = totalRows;
    size_t blockSize = (totalRows + nBlocks - 1) / nBlocks;
    for (size_t i = 0; i < nBlocks; i++) {
        size_t rowStart = i * rowsPerBlock;
        size_t rowEnd = std::min(rowStart + rowsPerBlock, totalRows);
        size_t rowsToRead = std::min(blockSize, remainingRows);
        size_t nLoaded = dataSource.loadDataBlock(rowsToRead);
        remainingRows -= nLoaded;
        if (rowStart >= totalRows)
            break;
        CSRNumericTablePtr dataTable = splitCSRBlock<algorithmFPType>(fullData, rowStart, rowEnd);
        NumericTablePtr blockTable = dataSource.getNumericTable();
        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(classifier::training::data, dataTable);
        algorithm.input.set(classifier::training::labels, blockTable);

        /* Build the Naive Bayes model */
        algorithm.compute();
    }
    /* Finalize the Naive Bayes model */
    algorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Read testDatasetFileName and create a numeric table to store the input data */
    testData = CSRNumericTablePtr(createSparseTable<float>(testDatasetFileName));

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
