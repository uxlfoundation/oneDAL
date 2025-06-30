/* file: lin_reg_qr_dense_distr.cpp */
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
!    C++ example of multiple linear regression in the distributed processing
!    mode.
!
!    The program trains the multiple linear regression model on a training
!    datasetFileName with a QR decomposition-based method and computes
!    regression for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LINEAR_REGRESSION_QR_DISTRIBUTED"></a>
 * \example lin_reg_qr_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::linear_regression;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const std::string trainDatasetFileNames[] = { "../data/distributed/linear_regression_train_1.csv",
                                              "../data/distributed/linear_regression_train_2.csv",
                                              "../data/distributed/linear_regression_train_3.csv",
                                              "../data/distributed/linear_regression_train_4.csv" };
const std::string testDatasetFileName = "../data/distributed/linear_regression_test.csv";

const size_t nBlocks = 4;

const size_t nFeatures = 10; /* Number of features in training and testing data sets */
const size_t nDependentVariables =
    2; /* Number of dependent variables that correspond to each observation */

void trainModel();
void testModel();

training::ResultPtr trainingResult;
prediction::ResultPtr predictionResult;

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   5,
                   &testDatasetFileName,
                   &trainDatasetFileNames[0],
                   &trainDatasetFileNames[1],
                   &trainDatasetFileNames[2],
                   &trainDatasetFileNames[3]);

    trainModel();
    testModel();

    return 0;
}

void trainModel() {
    /* Create an algorithm object to build the final multiple linear regression model on the master node */
    training::Distributed<step2Master, algorithmFPType, training::qrDense> masterAlgorithm;

    for (size_t i = 0; i < nBlocks; i++) {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileNames[i],
                                                          DataSource::notAllocateNumericTable,
                                                          DataSource::doDictionaryFromContext);

        /* Create Numeric Tables for training data and dependent variables */
        NumericTablePtr trainData(
            new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
        NumericTablePtr trainDependentVariables(
            new HomogenNumericTable<>(nDependentVariables, 0, NumericTable::doNotAllocate));
        NumericTablePtr mergedData(new MergedNumericTable(trainData, trainDependentVariables));

        /* Retrieve the data from input file */
        trainDataSource.loadDataBlock(mergedData.get());

        /* Create an algorithm object to train the multiple linear regression model based on the local-node data */
        training::Distributed<step1Local, algorithmFPType, training::qrDense> localAlgorithm;

        /* Pass a training data set and dependent values to the algorithm */
        localAlgorithm.input.set(training::data, trainData);
        localAlgorithm.input.set(training::dependentVariables, trainDependentVariables);

        /* Train the multiple linear regression model on the local-node data */
        localAlgorithm.compute();

        /* Set the local multiple linear regression model as input for the master-node algorithm */
        masterAlgorithm.input.add(training::partialModels, localAlgorithm.getPartialResult());
    }

    /* Merge and finalize the multiple linear regression model on the master node */
    masterAlgorithm.compute();

    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    trainingResult = masterAlgorithm.getResult();
    printNumericTable(trainingResult->get(training::model)->getBeta(),
                      "Linear Regression coefficients:");
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr testGroundTruth(
        new HomogenNumericTable<>(nDependentVariables, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Load the data from the data file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict values of multiple linear regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testData);
    algorithm.input.set(prediction::model, trainingResult->get(training::model));

    /* Predict values of multiple linear regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction),
                      "Linear Regression prediction results: (first 10 rows):",
                      10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}
