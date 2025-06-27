/* file: impl_als_csr_distr.cpp */
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
!    C++ example of the implicit alternating least squares (ALS) algorithm in
!    the distributed processing mode.
!
!    The program trains the implicit ALS model on a training data set.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-IMPLICIT_ALS_CSR_DISTRIBUTED"></a>
 * \example impl_als_csr_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::implicit_als;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string trainDatasetFileNames[nBlocks] = {
    "../data/distributed/implicit_als_trans_csr_1.csv",
    "../data/distributed/implicit_als_trans_csr_2.csv",
    "../data/distributed/implicit_als_trans_csr_3.csv",
    "../data/distributed/implicit_als_trans_csr_4.csv"
};

static int usersPartition[] = { nBlocks };

NumericTablePtr userOffsets[nBlocks];
NumericTablePtr itemOffsets[nBlocks];

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Algorithm parameters */
const size_t nUsers = 46; /* Full number of users */
const size_t nFactors = 2; /* Number of factors */
const size_t maxIterations = 5; /* Number of iterations in the implicit ALS training algorithm */

CSRNumericTablePtr dataTable[nBlocks];
CSRNumericTablePtr transposedDataTable[nBlocks];

NumericTablePtr predictedRatings[nBlocks][nBlocks];

KeyValueDataCollectionPtr userStep3LocalInput[nBlocks];
KeyValueDataCollectionPtr itemStep3LocalInput[nBlocks];

training::DistributedPartialResultStep4Ptr itemsPartialResultLocal[nBlocks];
training::DistributedPartialResultStep4Ptr usersPartialResultLocal[nBlocks];

void initializeModel();
void readData(size_t block);
void trainModel();
void testModel();
void printResults();

int main(int argc, char *argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &trainDatasetFileNames[0],
                   &trainDatasetFileNames[1],
                   &trainDatasetFileNames[2],
                   &trainDatasetFileNames[3]);

    for (size_t i = 0; i < nBlocks; i++) {
        readData(i);
    }

    initializeModel();

    trainModel();

    testModel();

    printResults();

    return 0;
}

KeyValueDataCollectionPtr initializeStep1Local(size_t block) {
    /* Create an algorithm object to perform the first step of the implicit ALS initialization algorithm */
    training::init::Distributed<step1Local, algorithmFPType, training::init::fastCSR> initAlgorithm;

    /* Set parameters of the algorithm */
    initAlgorithm.parameter.fullNUsers = nUsers;
    initAlgorithm.parameter.nFactors = nFactors;
    initAlgorithm.parameter.seed += block;
    initAlgorithm.parameter.partition.reset(
        new HomogenNumericTable<int>((int *)usersPartition, 1, 1));

    /* Pass a training data set to the algorithm */
    initAlgorithm.input.set(training::init::data, dataTable[block]);

    /* Compute partial results of the first step on local nodes */
    initAlgorithm.compute();

    training::init::PartialResultPtr partialResult = initAlgorithm.getPartialResult();
    itemStep3LocalInput[block] = partialResult->get(training::init::outputOfInitForComputeStep3);
    userOffsets[block] = partialResult->get(training::init::offsets, block);
    PartialModelPtr partialModelLocal = partialResult->get(training::init::partialModel);

    itemsPartialResultLocal[block].reset(new training::DistributedPartialResultStep4());
    itemsPartialResultLocal[block]->set(training::outputOfStep4ForStep1, partialModelLocal);

    return partialResult->get(training::init::outputOfStep1ForStep2);
}

void initializeStep2Local(size_t block, KeyValueDataCollectionPtr initStep2LocalInput) {
    /* Create an algorithm object to perform the second step of the implicit ALS initialization algorithm */
    training::init::Distributed<step2Local, algorithmFPType, training::init::fastCSR> initAlgorithm;

    initAlgorithm.input.set(training::init::inputOfStep2FromStep1, initStep2LocalInput);

    /* Compute partial results of the second step on local nodes */
    initAlgorithm.compute();

    training::init::DistributedPartialResultStep2Ptr partialResult =
        initAlgorithm.getPartialResult();
    transposedDataTable[block] =
        CSRNumericTable::cast(partialResult->get(training::init::transposedData));
    userStep3LocalInput[block] = partialResult->get(training::init::outputOfInitForComputeStep3);
    itemOffsets[block] = partialResult->get(training::init::offsets, block);
}

void initializeModel() {
    KeyValueDataCollectionPtr initStep1LocalResult[nBlocks];
    for (size_t i = 0; i < nBlocks; i++) {
        initStep1LocalResult[i] = initializeStep1Local(i);
    }

    /* Prepare input objects for the second step of the distributed initialization algorithm */
    KeyValueDataCollectionPtr initStep2LocalInput[nBlocks];
    for (size_t i = 0; i < nBlocks; i++) {
        initStep2LocalInput[i].reset(new KeyValueDataCollection());
        for (size_t j = 0; j < nBlocks; j++) {
            (*initStep2LocalInput[i])[j] = (*initStep1LocalResult[j])[i];
        }
    }
    for (size_t i = 0; i < nBlocks; i++) {
        initializeStep2Local(i, initStep2LocalInput[i]);
    }
}

training::DistributedPartialResultStep1Ptr computeStep1Local(
    const training::DistributedPartialResultStep4Ptr &partialResultLocal) {
    /* Create an algorithm object to perform first step of the implicit ALS training algorithm on local-node data */
    training::Distributed<step1Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModel,
                        partialResultLocal->get(training::outputOfStep4ForStep1));

    /* Compute partial results of the first step on local nodes */
    algorithm.compute();

    /* Get the computed partial results */
    return algorithm.getPartialResult();
}

NumericTablePtr computeStep2Master(
    const training::DistributedPartialResultStep1Ptr *step1LocalResult) {
    /* Create an algorithm object to perform second step of the implicit ALS training algorithm */
    training::Distributed<step2Master> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set the partial results of the first local step of distributed computations
       as input for the master-node algorithm */
    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(training::inputOfStep2FromStep1, step1LocalResult[i]);
    }

    /* Compute a partial result on the master node from the partial results on local nodes */
    algorithm.compute();

    /* Get the computed partial results */
    return algorithm.getPartialResult()->get(training::outputOfStep2ForStep4);
}

KeyValueDataCollectionPtr computeStep3Local(
    const NumericTablePtr &offsetTable,
    const training::DistributedPartialResultStep4Ptr &partialResultLocal,
    const KeyValueDataCollectionPtr &step3LocalInput) {
    /* Create an algorithm object to perform third step of the implicit ALS training algorithm on local-node data */
    training::Distributed<step3Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModel,
                        partialResultLocal->get(training::outputOfStep4ForStep3));
    algorithm.input.set(training::inputOfStep3FromInit, step3LocalInput);
    algorithm.input.set(training::offset, offsetTable);

    /* Compute partial results of the third step on local nodes */
    algorithm.compute();

    /* Get the computed partial results */
    return algorithm.getPartialResult()->get(training::outputOfStep3ForStep4);
}

training::DistributedPartialResultStep4Ptr computeStep4Local(
    const CSRNumericTablePtr &dataTable,
    const NumericTablePtr &step2MasterResult,
    const KeyValueDataCollectionPtr &step4LocalInput) {
    /* Create an algorithm object to perform fourth step of the implicit ALS training algorithm on local-node data */
    training::Distributed<step4Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModels, step4LocalInput);
    algorithm.input.set(training::partialData, dataTable);
    algorithm.input.set(training::inputOfStep4FromStep2, step2MasterResult);

    /* Build the implicit ALS partial model on the local node */
    algorithm.compute();

    /* Get the local implicit ALS partial models */
    return algorithm.getPartialResult();
}

void trainModel() {
    training::DistributedPartialResultStep1Ptr step1LocalResult[nBlocks];
    NumericTablePtr step2MasterResult;
    KeyValueDataCollectionPtr step3LocalResult[nBlocks];
    KeyValueDataCollectionPtr step4LocalInput[nBlocks];

    for (size_t i = 0; i < nBlocks; i++) {
        step4LocalInput[i].reset(new KeyValueDataCollection());
    }
    for (size_t iteration = 0; iteration < maxIterations; iteration++) {
        /* Update partial users factors */
        for (size_t i = 0; i < nBlocks; i++) {
            step1LocalResult[i] = computeStep1Local(itemsPartialResultLocal[i]);
        }
        step2MasterResult = computeStep2Master(step1LocalResult);

        for (size_t i = 0; i < nBlocks; i++) {
            step3LocalResult[i] = computeStep3Local(itemOffsets[i],
                                                    itemsPartialResultLocal[i],
                                                    itemStep3LocalInput[i]);
        }

        /* Prepare input objects for the fourth step of the distributed algorithm */
        for (size_t i = 0; i < nBlocks; i++) {
            for (size_t j = 0; j < nBlocks; j++) {
                (*step4LocalInput[i])[j] = (*step3LocalResult[j])[i];
            }
        }

        for (size_t i = 0; i < nBlocks; i++) {
            usersPartialResultLocal[i] =
                computeStep4Local(transposedDataTable[i], step2MasterResult, step4LocalInput[i]);
        }

        /* Update partial items factors */
        for (size_t i = 0; i < nBlocks; i++) {
            step1LocalResult[i] = computeStep1Local(usersPartialResultLocal[i]);
        }
        step2MasterResult = computeStep2Master(step1LocalResult);

        for (size_t i = 0; i < nBlocks; i++) {
            step3LocalResult[i] = computeStep3Local(userOffsets[i],
                                                    usersPartialResultLocal[i],
                                                    userStep3LocalInput[i]);
        }

        /* Prepare input objects for the fourth step of the distributed algorithm */
        for (size_t i = 0; i < nBlocks; i++) {
            for (size_t j = 0; j < nBlocks; j++) {
                (*step4LocalInput[i])[j] = (*step3LocalResult[j])[i];
            }
        }

        for (size_t i = 0; i < nBlocks; i++) {
            itemsPartialResultLocal[i] =
                computeStep4Local(dataTable[i], step2MasterResult, step4LocalInput[i]);
        }
    }
}

void testModel() {
    for (size_t i = 0; i < nBlocks; i++) {
        for (size_t j = 0; j < nBlocks; j++) {
            /* Create an algorithm object to predict ratings based in the implicit ALS partial models */
            prediction::ratings::Distributed<step1Local> algorithm;
            algorithm.parameter.nFactors = nFactors;

            /* Set input objects for the algorithm */
            algorithm.input.set(prediction::ratings::usersPartialModel,
                                usersPartialResultLocal[i]->get(training::outputOfStep4));
            algorithm.input.set(prediction::ratings::itemsPartialModel,
                                itemsPartialResultLocal[j]->get(training::outputOfStep4));

            /* Predict ratings */
            algorithm.compute();

            /* Retrieve the algorithm results */
            predictedRatings[i][j] = algorithm.getResult()->get(prediction::ratings::prediction);
        }
    }
}

void readData(size_t block) {
    /* Read trainDatasetFileName from a file and create a numeric table to store the input data */
    dataTable[block] = CSRNumericTablePtr(createSparseTable<float>(trainDatasetFileNames[block]));
}

void printResults() {
    for (size_t i = 0; i < nBlocks; i++) {
        for (size_t j = 0; j < nBlocks; j++) {
            std::cout << "Ratings for users block " << i << ", items block " << j << " :"
                      << std::endl;
            printALSRatings(userOffsets[i], itemOffsets[j], predictedRatings[i][j]);
        }
    }
}
