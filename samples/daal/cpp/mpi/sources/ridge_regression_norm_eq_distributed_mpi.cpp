/* file: ridge_regression_norm_eq_distributed_mpi.cpp */
/*******************************************************************************
* Copyright 2017 Intel Corporation
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
!    C++ sample of ridge regression in the distributed processing mode.
!
!    The program trains the ridge regression model on a training data set with the normal equations method and computes regression for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-RIDGE_REGRESSION_NORM_EQ_DISTRIBUTED"></a>
 * \example ridge_regression_norm_eq_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms::ridge_regression;

const std::string trainDatasetFileName = "data/linear_regression_train_data.csv";
const std::string trainDatasetLabelFileName = "data/linear_regression_train_responses.csv";
const std::string testDatasetFileName = "data/linear_regression_test_data.csv";
const std::string testDatasetLabelFileName = "data/linear_regression_test_responses.csv";

size_t nBlocks;

const size_t nDependentVariables =
    2; /* Number of dependent variables that correspond to each observation */

int rankId, comm_size;
#define mpi_root 0

void trainModel();
void testModel();

training::ResultPtr trainingResult;
prediction::ResultPtr predictionResult;

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &trainDatasetFileName,
                   &trainDatasetLabelFileName,
                   &testDatasetFileName,
                   &testDatasetLabelFileName);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    nBlocks = comm_size;
    trainModel();

    if (rankId == mpi_root) {
        testModel();
    }

    MPI_Finalize();

    return 0;
}

void trainModel() {
    size_t totalRows = countRowsCSV(trainDatasetFileName);
    size_t rowsPerRank = (totalRows + comm_size - 1) / comm_size;

    size_t rowStart = rankId * rowsPerRank;
    size_t rowEnd = std::min(rowStart + rowsPerRank, totalRows);

    if (rowStart >= totalRows)
        return;
    // Load only this rank's block
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    FileDataSource<CSVFeatureManager> trainLabelSource(trainDatasetLabelFileName,
                                                       DataSource::doAllocateNumericTable,
                                                       DataSource::doDictionaryFromContext);
    // Skip rows before rowStart
    size_t skip = rowStart;
    while (skip > 0) {
        size_t s1 = trainDataSource.loadDataBlock(skip);
        size_t s2 = trainLabelSource.loadDataBlock(skip);
        if (s1 == 0 || s2 == 0)
            break;
        skip -= s1;
    }

    size_t rowsToRead = rowEnd - rowStart;
    // Load rows for this rank
    trainDataSource.loadDataBlock(rowsToRead);
    trainLabelSource.loadDataBlock(rowsToRead);

    NumericTablePtr trainData = trainDataSource.getNumericTable();
    NumericTablePtr trainLabels = trainLabelSource.getNumericTable();

    training::Distributed<step1Local> localAlgorithm;
    localAlgorithm.input.set(training::data, trainData);
    localAlgorithm.input.set(training::dependentVariables, trainLabels);

    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (rankId == mpi_root) {
        serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);
    }

    byte* nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    MPI_Gather(nodeResults,
               perNodeArchLength,
               MPI_CHAR,
               serializedData.get(),
               perNodeArchLength,
               MPI_CHAR,
               mpi_root,
               MPI_COMM_WORLD);

    delete[] nodeResults;

    if (rankId == mpi_root) {
        /* Create an algorithm object to build the final ridge regression model on the master node */
        training::Distributed<step2Master> masterAlgorithm;

        for (size_t i = 0; i < nBlocks; ++i) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i,
                                       perNodeArchLength);

            training::PartialResultPtr dataForStep2FromStep1 =
                training::PartialResultPtr(new training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local ridge regression model as input for the master-node algorithm */
            masterAlgorithm.input.add(training::partialModels, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the ridge regression model on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        trainingResult = masterAlgorithm.getResult();
        printNumericTable(trainingResult->get(training::model)->getBeta(),
                          "Ridge Regression coefficients:");
    }
}

void testModel() {
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
    /* Create an algorithm object to predict values of multiple linear regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testDataSource.getNumericTable());
    algorithm.input.set(prediction::model, trainingResult->get(training::model));

    /* Predict values of multiple linear regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(prediction::prediction),
                      "Ridge Regression prediction results: (first 10 rows):",
                      10);
    printNumericTable(testLabelSource.getNumericTable(), "Ground truth (first 10 rows):", 10);
}
