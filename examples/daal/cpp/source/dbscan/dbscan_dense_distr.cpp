/* file: dbscan_dense_distr.cpp */
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
!    C++ example of dense DBSCAN clustering in the distributed processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DBSCAN_DENSE_DISTRIBUTED"></a>
 * \example dbscan_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string dataFileNames[nBlocks] = { "../data/distributed/dbscan_dense_1.csv",
                                             "../data/distributed/dbscan_dense_2.csv",
                                             "../data/distributed/dbscan_dense_3.csv",
                                             "../data/distributed/dbscan_dense_4.csv" };

typedef float algorithmFPType; /* Algorithm floating-point type */

/* Algorithm parameters */
const float epsilon = 0.02f;
const size_t minObservations = 180;

NumericTablePtr dataTable[nBlocks];

DataCollectionPtr partitionedData[nBlocks];
DataCollectionPtr partitionedPartialOrders[nBlocks];

DataCollectionPtr partialSplits[nBlocks];
DataCollectionPtr partialBoundingBoxes[nBlocks];
DataCollectionPtr newPartitionedData[nBlocks];
DataCollectionPtr newPartitionedDataIndices[nBlocks];
DataCollectionPtr newPartitionedPartialOrders[nBlocks];

DataCollectionPtr haloData[nBlocks];
DataCollectionPtr haloDataIndices[nBlocks];
DataCollectionPtr haloBlocks[nBlocks];

DataCollectionPtr queries[nBlocks];
DataCollectionPtr newQueries[nBlocks];

DataCollectionPtr assignmentQueries[nBlocks];
DataCollectionPtr newAssignmentQueries[nBlocks];

NumericTablePtr clusterStructure[nBlocks];
NumericTablePtr finishedFlag[nBlocks];
NumericTablePtr nClusters[nBlocks];
NumericTablePtr clusterOffset[nBlocks];
NumericTablePtr assignments[nBlocks];
NumericTablePtr totalNClusters;

void readData(size_t block);
void geometricPartitioning();
void clustering();
void printResults();

int computeFinishedFlag();

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &dataFileNames[0],
                   &dataFileNames[1],
                   &dataFileNames[2],
                   &dataFileNames[3]);

    for (size_t i = 0; i < nBlocks; i++) {
        readData(i);
    }

    geometricPartitioning();

    clustering();

    printResults();

    return 0;
}

void geometricPartitioning() {
    for (size_t block = 0; block < nBlocks; block++) {
        dbscan::Distributed<step1Local, algorithmFPType, dbscan::defaultDense> step1(block,
                                                                                     nBlocks);
        step1.input.set(dbscan::step1Data, dataTable[block]);
        step1.compute();

        partitionedData[block] = DataCollectionPtr(new DataCollection());
        partitionedPartialOrders[block] = DataCollectionPtr(new DataCollection());

        partitionedData[block]->push_back(dataTable[block]);
        partitionedPartialOrders[block]->push_back(
            step1.getPartialResult()->get(dbscan::partialOrder));
    }

    std::vector<std::pair<size_t, size_t> > coms;
    coms.push_back(std::make_pair(0, nBlocks));

    while (!coms.empty()) {
        std::vector<std::pair<size_t, size_t> > newComs;

        for (size_t comId = 0; comId < coms.size(); comId++) {
            const size_t beginBlock = coms[comId].first;
            const size_t endBlock = coms[comId].second;
            const size_t curNBlocks = endBlock - beginBlock;

            if (curNBlocks == 1) {
                continue;
            }

            for (size_t block = 0; block < curNBlocks; block++) {
                partialSplits[block + beginBlock] = DataCollectionPtr(new DataCollection());
                partialBoundingBoxes[block + beginBlock] = DataCollectionPtr(new DataCollection());
                newPartitionedData[block + beginBlock] = DataCollectionPtr(new DataCollection());
                newPartitionedPartialOrders[block + beginBlock] =
                    DataCollectionPtr(new DataCollection());
            }

            for (size_t block = 0; block < curNBlocks; block++) {
                dbscan::Distributed<step2Local, algorithmFPType, dbscan::defaultDense> step2(
                    block,
                    curNBlocks);
                step2.input.set(dbscan::partialData, partitionedData[block + beginBlock]);
                step2.compute();
                NumericTablePtr curBoundingBox = step2.getPartialResult()->get(dbscan::boundingBox);

                for (size_t destBlock = 0; destBlock < curNBlocks; destBlock++) {
                    partialBoundingBoxes[destBlock + beginBlock]->push_back(curBoundingBox);
                }
            }

            const size_t leftBlocks = curNBlocks / 2;
            const size_t rightBlocks = curNBlocks - leftBlocks;

            for (size_t block = 0; block < curNBlocks; block++) {
                dbscan::Distributed<step3Local, algorithmFPType, dbscan::defaultDense> step3(
                    leftBlocks,
                    rightBlocks);
                step3.input.set(dbscan::partialData, partitionedData[block + beginBlock]);
                step3.input.set(dbscan::step3PartialBoundingBoxes,
                                partialBoundingBoxes[block + beginBlock]);
                step3.compute();
                NumericTablePtr curSplit = step3.getPartialResult()->get(dbscan::split);

                for (size_t destBlock = 0; destBlock < curNBlocks; destBlock++) {
                    partialSplits[destBlock + beginBlock]->push_back(curSplit);
                }
            }

            for (size_t block = 0; block < curNBlocks; block++) {
                dbscan::Distributed<step4Local, algorithmFPType, dbscan::defaultDense> step4(
                    leftBlocks,
                    rightBlocks);
                step4.input.set(dbscan::partialData, partitionedData[block + beginBlock]);
                step4.input.set(dbscan::step4PartialOrders,
                                partitionedPartialOrders[block + beginBlock]);
                step4.input.set(dbscan::step4PartialSplits, partialSplits[block + beginBlock]);
                step4.compute();

                DataCollectionPtr curPartitionedData =
                    step4.getPartialResult()->get(dbscan::partitionedData);
                DataCollectionPtr curPartitionedPartialOrders =
                    step4.getPartialResult()->get(dbscan::partitionedPartialOrders);

                for (size_t destBlock = 0; destBlock < curNBlocks; destBlock++) {
                    newPartitionedData[destBlock + beginBlock]->push_back(
                        (*curPartitionedData)[destBlock]);
                    newPartitionedPartialOrders[destBlock + beginBlock]->push_back(
                        (*curPartitionedPartialOrders)[destBlock]);
                }
            }

            for (size_t block = 0; block < curNBlocks; block++) {
                partitionedData[block + beginBlock] = newPartitionedData[block + beginBlock];
                partitionedPartialOrders[block + beginBlock] =
                    newPartitionedPartialOrders[block + beginBlock];
            }

            newComs.push_back(std::make_pair(beginBlock, beginBlock + leftBlocks));
            newComs.push_back(std::make_pair(beginBlock + leftBlocks, endBlock));
        }

        coms = newComs;
    }
}

void clustering() {
    for (size_t block = 0; block < nBlocks; block++) {
        partialBoundingBoxes[block] = DataCollectionPtr(new DataCollection());
        haloData[block] = DataCollectionPtr(new DataCollection());
        haloDataIndices[block] = DataCollectionPtr(new DataCollection());
        haloBlocks[block] = DataCollectionPtr(new DataCollection());
    }

    for (size_t block = 0; block < nBlocks; block++) {
        dbscan::Distributed<step2Local, algorithmFPType, dbscan::defaultDense> step2(block,
                                                                                     nBlocks);
        step2.input.set(dbscan::partialData, partitionedData[block]);
        step2.compute();
        NumericTablePtr curBoundingBox = step2.getPartialResult()->get(dbscan::boundingBox);

        for (size_t destBlock = 0; destBlock < nBlocks; destBlock++) {
            partialBoundingBoxes[destBlock]->push_back(curBoundingBox);
        }
    }

    for (size_t block = 0; block < nBlocks; block++) {
        dbscan::Distributed<step5Local, algorithmFPType, dbscan::defaultDense> step5(block,
                                                                                     nBlocks,
                                                                                     epsilon);
        step5.input.set(dbscan::partialData, partitionedData[block]);
        step5.input.set(dbscan::step5PartialBoundingBoxes, partialBoundingBoxes[block]);
        step5.compute();
        DataCollectionPtr curHaloData = step5.getPartialResult()->get(dbscan::partitionedHaloData);
        DataCollectionPtr curHaloDataIndices =
            step5.getPartialResult()->get(dbscan::partitionedHaloDataIndices);

        for (size_t destBlock = 0; destBlock < nBlocks; destBlock++) {
            NumericTablePtr dataTable =
                services::staticPointerCast<NumericTable, SerializationIface>(
                    (*curHaloData)[destBlock]);
            NumericTablePtr dataIndicesTable =
                services::staticPointerCast<NumericTable, SerializationIface>(
                    (*curHaloDataIndices)[destBlock]);
            if (dataTable->getNumberOfRows() > 0) {
                haloData[destBlock]->push_back(dataTable);
                haloDataIndices[destBlock]->push_back(dataIndicesTable);
                haloBlocks[destBlock]->push_back(
                    HomogenNumericTable<int>::create(1,
                                                     1,
                                                     NumericTableIface::doAllocate,
                                                     (int)block));
            }
        }
    }

    for (size_t block = 0; block < nBlocks; block++) {
        queries[block] = DataCollectionPtr(new DataCollection());
    }

    for (size_t block = 0; block < nBlocks; block++) {
        dbscan::Distributed<step6Local, algorithmFPType, dbscan::defaultDense> step6(
            block,
            nBlocks,
            epsilon,
            minObservations);
        step6.parameter().memorySavingMode = false;

        step6.input.set(dbscan::partialData, partitionedData[block]);
        step6.input.set(dbscan::haloData, haloData[block]);
        step6.input.set(dbscan::haloDataIndices, haloDataIndices[block]);
        step6.input.set(dbscan::haloBlocks, haloBlocks[block]);
        step6.compute();
        clusterStructure[block] = step6.getPartialResult()->get(dbscan::step6ClusterStructure);
        finishedFlag[block] = step6.getPartialResult()->get(dbscan::step6FinishedFlag);
        nClusters[block] = step6.getPartialResult()->get(dbscan::step6NClusters);

        DataCollectionPtr curQueries = step6.getPartialResult()->get(dbscan::step6Queries);

        for (size_t destBlock = 0; destBlock < nBlocks; destBlock++) {
            NumericTablePtr table = services::staticPointerCast<NumericTable, SerializationIface>(
                (*curQueries)[destBlock]);
            if (table->getNumberOfRows() > 0) {
                queries[destBlock]->push_back(table);
            }
        }
    }

    while (computeFinishedFlag() == 0) {
        for (size_t block = 0; block < nBlocks; block++) {
            newQueries[block] = DataCollectionPtr(new DataCollection());
        }

        for (size_t block = 0; block < nBlocks; block++) {
            dbscan::Distributed<step8Local, algorithmFPType, dbscan::defaultDense> step8(block,
                                                                                         nBlocks);
            step8.input.set(dbscan::step8InputClusterStructure, clusterStructure[block]);
            step8.input.set(dbscan::step8InputNClusters, nClusters[block]);
            step8.input.set(dbscan::step8PartialQueries, queries[block]);
            step8.compute();

            clusterStructure[block] = step8.getPartialResult()->get(dbscan::step8ClusterStructure);
            finishedFlag[block] = step8.getPartialResult()->get(dbscan::step8FinishedFlag);
            nClusters[block] = step8.getPartialResult()->get(dbscan::step8NClusters);

            DataCollectionPtr curQueries = step8.getPartialResult()->get(dbscan::step8Queries);

            for (size_t destBlock = 0; destBlock < nBlocks; destBlock++) {
                NumericTablePtr table =
                    services::staticPointerCast<NumericTable, SerializationIface>(
                        (*curQueries)[destBlock]);
                if (table->getNumberOfRows() > 0) {
                    newQueries[destBlock]->push_back(table);
                }
            }
        }

        for (size_t block = 0; block < nBlocks; block++) {
            queries[block] = newQueries[block];
        }
    }

    {
        DataCollectionPtr partialNClusters(new DataCollection());
        for (size_t block = 0; block < nBlocks; block++) {
            partialNClusters->push_back(nClusters[block]);
        }

        dbscan::Distributed<step9Master, algorithmFPType, dbscan::defaultDense> step9;
        step9.input.set(dbscan::partialNClusters, partialNClusters);
        step9.compute();
        step9.finalizeCompute();

        totalNClusters = step9.getResult()->get(dbscan::step9NClusters);

        DataCollectionPtr curClusterOffsets = step9.getPartialResult()->get(dbscan::clusterOffsets);

        for (size_t block = 0; block < nBlocks; block++) {
            clusterOffset[block] = services::staticPointerCast<NumericTable, SerializationIface>(
                (*curClusterOffsets)[block]);
        }
    }

    for (size_t block = 0; block < nBlocks; block++) {
        queries[block] = DataCollectionPtr(new DataCollection());
    }

    for (size_t block = 0; block < nBlocks; block++) {
        dbscan::Distributed<step10Local, algorithmFPType, dbscan::defaultDense> step10(block,
                                                                                       nBlocks);
        step10.input.set(dbscan::step10InputClusterStructure, clusterStructure[block]);
        step10.input.set(dbscan::step10ClusterOffset, clusterOffset[block]);
        step10.compute();

        clusterStructure[block] = step10.getPartialResult()->get(dbscan::step10ClusterStructure);
        finishedFlag[block] = step10.getPartialResult()->get(dbscan::step10FinishedFlag);

        DataCollectionPtr curQueries = step10.getPartialResult()->get(dbscan::step10Queries);

        for (size_t destBlock = 0; destBlock < nBlocks; destBlock++) {
            NumericTablePtr table = services::staticPointerCast<NumericTable, SerializationIface>(
                (*curQueries)[destBlock]);
            if (table->getNumberOfRows() > 0) {
                queries[destBlock]->push_back(table);
            }
        }
    }

    while (computeFinishedFlag() == 0) {
        for (size_t block = 0; block < nBlocks; block++) {
            newQueries[block] = DataCollectionPtr(new DataCollection());
        }

        for (size_t block = 0; block < nBlocks; block++) {
            dbscan::Distributed<step11Local, algorithmFPType, dbscan::defaultDense> step11(block,
                                                                                           nBlocks);
            step11.input.set(dbscan::step11InputClusterStructure, clusterStructure[block]);
            step11.input.set(dbscan::step11PartialQueries, queries[block]);
            step11.compute();

            clusterStructure[block] =
                step11.getPartialResult()->get(dbscan::step11ClusterStructure);
            finishedFlag[block] = step11.getPartialResult()->get(dbscan::step11FinishedFlag);

            DataCollectionPtr curQueries = step11.getPartialResult()->get(dbscan::step11Queries);

            for (size_t destBlock = 0; destBlock < nBlocks; destBlock++) {
                NumericTablePtr table =
                    services::staticPointerCast<NumericTable, SerializationIface>(
                        (*curQueries)[destBlock]);
                if (table->getNumberOfRows() > 0) {
                    newQueries[destBlock]->push_back(table);
                }
            }
        }

        for (size_t block = 0; block < nBlocks; block++) {
            queries[block] = newQueries[block];
        }
    }

    for (size_t block = 0; block < nBlocks; block++) {
        assignmentQueries[block] = DataCollectionPtr(new DataCollection());
    }

    for (size_t block = 0; block < nBlocks; block++) {
        dbscan::Distributed<step12Local, algorithmFPType, dbscan::defaultDense> step12(block,
                                                                                       nBlocks);
        step12.input.set(dbscan::step12InputClusterStructure, clusterStructure[block]);
        step12.input.set(dbscan::step12PartialOrders, partitionedPartialOrders[block]);
        step12.compute();

        DataCollectionPtr curAssignmentQueries =
            step12.getPartialResult()->get(dbscan::assignmentQueries);

        for (size_t destBlock = 0; destBlock < nBlocks; destBlock++) {
            NumericTablePtr table = services::staticPointerCast<NumericTable, SerializationIface>(
                (*curAssignmentQueries)[destBlock]);
            if (table->getNumberOfRows() > 0) {
                assignmentQueries[destBlock]->push_back(table);
            }
        }
    }

    for (size_t block = 0; block < nBlocks; block++) {
        dbscan::Distributed<step13Local, algorithmFPType, dbscan::defaultDense> step13;
        step13.input.set(dbscan::partialAssignmentQueries, assignmentQueries[block]);
        step13.compute();
        step13.finalizeCompute();

        assignments[block] = step13.getResult()->get(dbscan::step13Assignments);
    }
}

int computeFinishedFlag() {
    DataCollectionPtr partialFinishedFlags(new DataCollection());

    for (size_t block = 0; block < nBlocks; block++) {
        partialFinishedFlags->push_back(finishedFlag[block]);
    }

    dbscan::Distributed<step7Master, algorithmFPType, dbscan::defaultDense> step7;
    step7.input.set(dbscan::partialFinishedFlags, partialFinishedFlags);
    step7.compute();
    NumericTablePtr finishedFlag = step7.getPartialResult()->get(dbscan::finishedFlag);

    int finishedFlagValue = finishedFlag->getValue<int>(0, 0);
    return finishedFlagValue;
}

void readData(size_t block) {
    /* Read trainDatasetFileName from a file and create a numeric table to store the input data */
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(dataFileNames[block],
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();
    dataTable[block] = dataSource.getNumericTable();
}

void printResults() {
    printNumericTable(totalNClusters, "Number of clusters:");
    for (size_t block = 0; block < nBlocks; block++) {
        printNumericTable(assignments[block],
                          "Assignments of first 20 observations from block:",
                          20);
    }
}
