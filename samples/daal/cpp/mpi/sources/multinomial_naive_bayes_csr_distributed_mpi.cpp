/* file: multinomial_naive_bayes_csr_distributed_mpi.cpp */
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
!    C++ sample of Naive Bayes classification in the distributed processing
!    mode.
!
!    The program trains the Naive Bayes model on a supplied training data set
!    and then performs classification of previously unseen data.
!******************************************************************************/

/**
 * <a name="DAAL-SAMPLE-CPP-MULTINOMIAL_NAIVE_BAYES_CSR_DISTRIBUTED"></a>
 * \example multinomial_naive_bayes_csr_distributed_mpi.cpp
 */

#include <mpi.h>
#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::multinomial_naive_bayes;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Input data set parameters */
const std::string trainDatasetFileName = "data/naivebayes_train_csr.csv";
const std::string trainGroundTruthFileName = "data/naivebayes_train_labels.csv";

std::string testdatasetFileName = "data/naivebayes_test_csr.csv";
std::string testGroundTruthFileName = "data/naivebayes_test_labels.csv";

const size_t nClasses = 20;
const size_t nBlocks = 4;

int rankId, comm_size;
#define mpi_root 0

void trainModel();
void testModel();
void printResults();

training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    trainModel();

    if (rankId == mpi_root) {
        testModel();
        printResults();
    }

    MPI_Finalize();

    return 0;
}

void trainModel() {
    /* Retrieve the input data from a .csv file */
    CSRNumericTable* trainDataTable = createSparseTable<float>(trainDatasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainLabelsSource(trainGroundTruthFileName,
                                                        DataSource::doAllocateNumericTable,
                                                        DataSource::doDictionaryFromContext);

    /* Retrieve the data from input files */
    trainLabelsSource.loadDataBlock();

    /* Create an algorithm object to train the Naive Bayes model based on the local-node data */
    training::Distributed<step1Local, algorithmFPType, training::fastCSR> localAlgorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(classifier::training::data, CSRNumericTablePtr(trainDataTable));
    localAlgorithm.input.set(classifier::training::labels, trainLabelsSource.getNumericTable());

    /* Train the Naive Bayes model on local nodes */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (rankId == mpi_root) {
        serializedData.reset(new byte[perNodeArchLength * nBlocks]);
    }

    {
        services::SharedPtr<byte> nodeResults(new byte[perNodeArchLength]);
        dataArch.copyArchiveToArray(nodeResults.get(), perNodeArchLength);

        /* Transfer partial results to step 2 on the root node */
        MPI_Gather(nodeResults.get(),
                   perNodeArchLength,
                   MPI_CHAR,
                   serializedData.get(),
                   perNodeArchLength,
                   MPI_CHAR,
                   mpi_root,
                   MPI_COMM_WORLD);
    }

    if (rankId == mpi_root) {
        /* Create an algorithm object to build the final Naive Bayes model on the master node */
        training::Distributed<step2Master, algorithmFPType, training::fastCSR> masterAlgorithm(
            nClasses);

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i,
                                       perNodeArchLength);

            training::PartialResultPtr dataForStep2FromStep1(new training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local Naive Bayes model as input for the master-node algorithm */
            masterAlgorithm.input.add(training::partialModels, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the Naive Bayes model on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        trainingResult = masterAlgorithm.getResult();
    }
}

void testModel() {
    /* Retrieve the input data from a .csv file */
    CSRNumericTable* testDataTable = createSparseTable<float>(testDatasetFileName);

    /* Create an algorithm object to predict values of the Naive Bayes model */
    prediction::Batch<algorithmFPType, prediction::fastCSR> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, CSRNumericTablePtr(testDataTable));
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Predict values of the Naive Bayes model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults() {
    FileDataSource<CSVFeatureManager> testGroundTruth(testGroundTruthFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testGroundTruth.loadDataBlock();

    printNumericTables<int, int>(testGroundTruth.getNumericTable().get(),
                                 predictionResult->get(classifier::prediction::prediction).get(),
                                 "Ground truth",
                                 "Classification results",
                                 "NaiveBayes classification results (first 20 observations):",
                                 20);
}
