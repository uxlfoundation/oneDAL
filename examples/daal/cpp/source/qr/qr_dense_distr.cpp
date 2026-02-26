/* file: qr_dense_distr.cpp */
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
!    C++ example of computing QR decomposition in the distributed processing
!    mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-QR_DISTRIBUTED"></a>
 * \example qr_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileName = { "data/qr.csv" };

void computestep1Local(size_t block, const NumericTablePtr &data);
void computeOnMasterNode();
void finalizeComputestep1Local(size_t block);

data_management::DataCollectionPtr dataFromStep1ForStep2[nBlocks];
data_management::DataCollectionPtr dataFromStep1ForStep3[nBlocks];
data_management::DataCollectionPtr dataFromStep2ForStep3[nBlocks];
NumericTablePtr R;
NumericTablePtr Qi[nBlocks];

int main(int argc, char *argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);
    size_t totalRows = countRowsCSV(datasetFileName);
    size_t blockSize = (totalRows + nBlocks - 1) / nBlocks;

    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
    size_t remainingRows = totalRows;

    for (size_t i = 0; i < nBlocks; i++) {
        size_t rowsToRead = std::min(blockSize, remainingRows);
        size_t nLoaded = dataSource.loadDataBlock(rowsToRead);
        remainingRows -= nLoaded;
        NumericTablePtr blockTable = dataSource.getNumericTable();
        computestep1Local(i, blockTable);
    }

    computeOnMasterNode();

    for (size_t i = 0; i < nBlocks; i++) {
        finalizeComputestep1Local(i);
    }

    /* Print the results */
    printNumericTable(Qi[0], "Part of orthogonal matrix Q from 1st node:", 10);
    printNumericTable(R, "Triangular matrix R:");

    return 0;
}

void computestep1Local(size_t block, const NumericTablePtr &data) {
    /* Create an algorithm to compute QR decomposition on the local node */
    qr::Distributed<step1Local> algorithm;

    algorithm.input.set(qr::data, data);

    /* Compute QR decomposition */
    algorithm.compute();

    dataFromStep1ForStep2[block] = algorithm.getPartialResult()->get(qr::outputOfStep1ForStep2);
    dataFromStep1ForStep3[block] = algorithm.getPartialResult()->get(qr::outputOfStep1ForStep3);
}

void computeOnMasterNode() {
    /* Create an algorithm to compute QR decomposition on the master node */
    qr::Distributed<step2Master> algorithm;

    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(qr::inputOfStep2FromStep1, i, dataFromStep1ForStep2[i]);
    }

    /* Compute QR decomposition */
    algorithm.compute();

    qr::DistributedPartialResultPtr pres = algorithm.getPartialResult();
    KeyValueDataCollectionPtr inputForStep3FromStep2 = pres->get(qr::outputOfStep2ForStep3);

    for (size_t i = 0; i < nBlocks; i++) {
        dataFromStep2ForStep3[i] =
            services::staticPointerCast<data_management::DataCollection, SerializationIface>(
                (*inputForStep3FromStep2)[i]);
    }

    qr::ResultPtr res = algorithm.getResult();

    R = res->get(qr::matrixR);
}

void finalizeComputestep1Local(size_t block) {
    /* Create an algorithm to compute QR decomposition on the master node */
    qr::Distributed<step3Local> algorithm;

    algorithm.input.set(qr::inputOfStep3FromStep1, dataFromStep1ForStep3[block]);
    algorithm.input.set(qr::inputOfStep3FromStep2, dataFromStep2ForStep3[block]);

    /* Compute QR decomposition */
    algorithm.compute();

    algorithm.finalizeCompute();

    qr::ResultPtr res = algorithm.getResult();

    Qi[block] = res->get(qr::matrixQ);
}
