/* file: kmeans_init_dense_distr.cpp */
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
!    C++ example of dense K-Means clustering in the distributed processing mode
!******************************************************************************/

/**
* <a name="DAAL-EXAMPLE-CPP-KMEANS_INIT_DENSE_DISTRIBUTED"></a>
* \example kmeans_init_dense_distr.cpp
*/

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

typedef float algorithmFPType; /* Algorithm floating-point type */

/* K-Means algorithm parameters */
const size_t nClusters = 20;
const size_t nIterations = 5;
const size_t nBlocks = 4;
const size_t nVectorsInBlock = 2500;

const std::string dataFileNames[] = { "../data/distributed/kmeans_dense_1.csv",
                                      "../data/distributed/kmeans_dense_2.csv",
                                      "../data/distributed/kmeans_dense_3.csv",
                                      "../data/distributed/kmeans_dense_4.csv" };

void loadData(NumericTablePtr data[nBlocks]) {
    for (size_t i = 0; i < nBlocks; i++) {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(dataFileNames[i],
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock();
        data[i] = dataSource.getNumericTable();
    }
}

template <kmeans::init::Method method>
NumericTablePtr initCentroids(const NumericTablePtr data[nBlocks]);

void calculateCentroids(const NumericTablePtr& initialCentroids,
                        const NumericTablePtr data[nBlocks]);

template <kmeans::init::Method method>
void runKMeans(const NumericTablePtr data[nBlocks], const char* methodName) {
    std::cout << "K-means init parameters: method = " << methodName << std::endl;
    NumericTablePtr centroids = initCentroids<method>(data);
    calculateCentroids(centroids, data);
}

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &dataFileNames[0],
                   &dataFileNames[1],
                   &dataFileNames[2],
                   &dataFileNames[3]);

    NumericTablePtr data[nBlocks];
    loadData(data);

    runKMeans<kmeans::init::plusPlusDense>(data, "plusPlusDense");
    runKMeans<kmeans::init::parallelPlusDense>(data, "parallelPlusDense");

    return 0;
}

template <kmeans::init::Method method>
NumericTablePtr initStep1(const NumericTablePtr data[nBlocks]) {
    for (size_t i = 0; i < nBlocks; i++) {
        /* Create an algorithm object for the K-Means algorithm */
        kmeans::init::Distributed<step1Local, algorithmFPType, method> local(
            nClusters,
            nBlocks * nVectorsInBlock,
            i * nVectorsInBlock);
        local.input.set(kmeans::init::data, data[i]);
        local.compute();
        NumericTablePtr pNewCenters = local.getPartialResult()->get(kmeans::init::partialCentroids);
        if (pNewCenters.get())
            return pNewCenters;
    }
    return NumericTablePtr();
}

template <kmeans::init::Method method>
void initStep23(const NumericTablePtr data[nBlocks],
                DataCollectionPtr localNodeData[nBlocks],
                const NumericTablePtr& step2Input,
                kmeans::init::Distributed<step3Master, algorithmFPType, method>& step3,
                bool bFirstIteration) {
    for (size_t i = 0; i < nBlocks; i++) {
        kmeans::init::Distributed<step2Local, algorithmFPType, method> step2(nClusters,
                                                                             bFirstIteration);
        step2.input.set(kmeans::init::data, data[i]);
        step2.input.set(kmeans::init::internalInput, localNodeData[i]);
        step2.input.set(kmeans::init::inputOfStep2, step2Input);
        step2.compute();
        if (bFirstIteration)
            localNodeData[i] = step2.getPartialResult()->get(kmeans::init::internalResult);
        step3.input.add(kmeans::init::inputOfStep3FromStep2,
                        i,
                        step2.getPartialResult()->get(kmeans::init::outputOfStep2ForStep3));
    }
    step3.compute();
}

template <kmeans::init::Method method>
NumericTablePtr initStep4(const NumericTablePtr data[nBlocks],
                          DataCollectionPtr localNodeData[nBlocks],
                          kmeans::init::Distributed<step3Master, algorithmFPType, method>& step3) {
    std::vector<NumericTablePtr> aRes;
    for (size_t i = 0; i < nBlocks; ++i) {
        /* Get an input for step 4 on this node if any */
        NumericTablePtr step3Output =
            step3.getPartialResult()->get(kmeans::init::outputOfStep3ForStep4, i);
        if (!step3Output)
            continue; /* can be null */

        /* Create an algorithm object for the step 4 */
        kmeans::init::Distributed<step4Local, algorithmFPType, method> step4(nClusters);
        /* Set the input data to the algorithm */
        step4.input.set(kmeans::init::data, data[i]);
        step4.input.set(kmeans::init::internalInput, localNodeData[i]);
        step4.input.set(kmeans::init::inputOfStep4FromStep3, step3Output);
        /* Compute and get the result */
        step4.compute();
        aRes.push_back(step4.getPartialResult()->get(kmeans::init::outputOfStep4));
    }
    if (!aRes.size())
        return NumericTablePtr();
    if (aRes.size() == 1)
        return aRes[0];
    /* For parallelPlus algorithm */
    RowMergedNumericTablePtr pMerged(new RowMergedNumericTable());
    for (size_t i = 0; i < aRes.size(); ++i)
        pMerged->addNumericTable(aRes[i]);
    return NumericTable::cast(pMerged);
}

template <>
NumericTablePtr initCentroids<kmeans::init::plusPlusDense>(const NumericTablePtr data[nBlocks]) {
    /* Internal data to be stored on the local nodes */
    DataCollectionPtr localNodeData[nBlocks];
    /* Numeric table to collect the results */
    RowMergedNumericTablePtr pCentroids(new RowMergedNumericTable());
    /* First step on the local nodes */
    NumericTablePtr pNewCentroids = initStep1<kmeans::init::plusPlusDense>(data);
    pCentroids->addNumericTable(pNewCentroids);

    /* Create an algorithm object for the step 3 */
    kmeans::init::Distributed<step3Master, algorithmFPType, kmeans::init::plusPlusDense> step3(
        nClusters);
    for (size_t iCenter = 1; iCenter < nClusters; ++iCenter) {
        /* Perform steps 2 and 3 */
        initStep23<kmeans::init::plusPlusDense>(data,
                                                localNodeData,
                                                pNewCentroids,
                                                step3,
                                                iCenter == 1);
        /* Perform steps 4 */
        pNewCentroids = initStep4<kmeans::init::plusPlusDense>(data, localNodeData, step3);
        pCentroids->addNumericTable(pNewCentroids);
    }
    return NumericTable::cast(pCentroids);
}

template <>
NumericTablePtr initCentroids<kmeans::init::parallelPlusDense>(
    const NumericTablePtr data[nBlocks]) {
    /* Internal data to be stored on the local nodes */
    DataCollectionPtr localNodeData[nBlocks];
    /* First step on the local nodes */
    NumericTablePtr pNewCentroids = initStep1<kmeans::init::parallelPlusDense>(data);

    /* Create an algorithm object for the step 5 */
    kmeans::init::Distributed<step5Master, algorithmFPType, kmeans::init::parallelPlusDense> step5(
        nClusters);
    step5.input.add(kmeans::init::inputCentroids, pNewCentroids);
    /* Create an algorithm object for the step 3 */
    kmeans::init::Distributed<step3Master, algorithmFPType, kmeans::init::parallelPlusDense> step3(
        nClusters);
    for (size_t iRound = 0; iRound < step5.parameter.nRounds; ++iRound) {
        /* Perform steps 2 and 3 */
        initStep23<kmeans::init::parallelPlusDense>(data,
                                                    localNodeData,
                                                    pNewCentroids,
                                                    step3,
                                                    iRound == 0);
        /* Perform step 4 */
        pNewCentroids = initStep4<kmeans::init::parallelPlusDense>(data, localNodeData, step3);
        step5.input.add(kmeans::init::inputCentroids, pNewCentroids);
    }
    /* One more step 2 */
    for (size_t i = 0; i < nBlocks; i++) {
        /* Create an algorithm object for the step 2 */
        kmeans::init::Distributed<step2Local, algorithmFPType, kmeans::init::parallelPlusDense>
            local(nClusters, false);
        local.parameter.outputForStep5Required = true;
        /* Set the input data to the algorithm */
        local.input.set(kmeans::init::data, data[i]);
        local.input.set(kmeans::init::internalInput, localNodeData[i]);
        local.input.set(kmeans::init::inputOfStep2, pNewCentroids);
        /* Compute and get the result */
        local.compute();
        /* Add the result to the input of step 5 */
        step5.input.add(kmeans::init::inputOfStep5FromStep2,
                        local.getPartialResult()->get(kmeans::init::outputOfStep2ForStep5));
    }
    step5.input.set(kmeans::init::inputOfStep5FromStep3,
                    step3.getPartialResult()->get(kmeans::init::outputOfStep3ForStep5));
    step5.compute();
    step5.finalizeCompute();
    return step5.getResult()->get(kmeans::init::centroids);
}

void calculateCentroids(const NumericTablePtr& initialCentroids,
                        const NumericTablePtr data[nBlocks]) {
    kmeans::Distributed<step2Master> masterAlgorithm(nClusters);

    NumericTablePtr assignments[nBlocks];
    NumericTablePtr centroids = initialCentroids;
    NumericTablePtr objectiveFunction;

    /* Calculate centroids */
    for (size_t it = 0; it < nIterations; it++) {
        for (size_t i = 0; i < nBlocks; i++) {
            /* Create an algorithm object for the K-Means algorithm */
            kmeans::Distributed<step1Local> localAlgorithm(nClusters, false);

            /* Set the input data to the algorithm */
            localAlgorithm.input.set(kmeans::data, data[i]);
            localAlgorithm.input.set(kmeans::inputCentroids, centroids);

            localAlgorithm.compute();
            masterAlgorithm.input.add(kmeans::partialResults, localAlgorithm.getPartialResult());
        }

        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        centroids = masterAlgorithm.getResult()->get(kmeans::centroids);
        objectiveFunction = masterAlgorithm.getResult()->get(kmeans::objectiveFunction);
    }

    /* Calculate assignments */
    for (size_t i = 0; i < nBlocks; i++) {
        /* Create an algorithm object for the K-Means algorithm */
        kmeans::Batch<> localAlgorithm(nClusters, 0);

        /* Set the input data to the algorithm */
        localAlgorithm.input.set(kmeans::data, data[i]);
        localAlgorithm.input.set(kmeans::inputCentroids, centroids);

        localAlgorithm.compute();

        assignments[i] = localAlgorithm.getResult()->get(kmeans::assignments);
    }

    /* Print the clusterization results */
    printNumericTable(assignments[0], "First 10 cluster assignments from 1st node:", 10);
    printNumericTable(centroids, "First 10 dimensions of centroids:", 20, 10);
    printNumericTable(objectiveFunction, "Objective function value:");
}
