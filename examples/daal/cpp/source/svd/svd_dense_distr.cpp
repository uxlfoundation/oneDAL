/* file: svd_dense_distr.cpp */
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
!    C++ example of singular value decomposition (SVD) in the distributed
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVD_DISTRIBUTED"></a>
 * \example svd_dense_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const size_t nBlocks = 4;

const std::string datasetFileNames[] = { "../data/distributed/svd_1.csv",
                                         "../data/distributed/svd_2.csv",
                                         "../data/distributed/svd_3.csv",
                                         "../data/distributed/svd_4.csv" };

void computestep1Local(size_t block);
void computeOnMasterNode();
void finalizeComputestep1Local(size_t block);

DataCollectionPtr dataFromStep1ForStep2[nBlocks];
DataCollectionPtr dataFromStep1ForStep3[nBlocks];
DataCollectionPtr dataFromStep2ForStep3[nBlocks];
NumericTablePtr Sigma;
NumericTablePtr V;
NumericTablePtr Ui[nBlocks];

int main(int argc, char* argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &datasetFileNames[0],
                   &datasetFileNames[1],
                   &datasetFileNames[2],
                   &datasetFileNames[3]);

    for (size_t i = 0; i < nBlocks; i++) {
        computestep1Local(i);
    }

    computeOnMasterNode();

    for (size_t i = 0; i < nBlocks; i++) {
        finalizeComputestep1Local(i);
    }

    /* Print the results */
    printNumericTable(Sigma, "Singular values:");
    printNumericTable(V, "Right orthogonal matrix V:");
    printNumericTable(Ui[0], "Part of left orthogonal matrix U from 1st node:", 10);

    return 0;
}

void computestep1Local(size_t block) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileNames[block],
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the input data */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute SVD on the local node */
    svd::Distributed<step1Local> algorithm;

    algorithm.input.set(svd::data, dataSource.getNumericTable());

    /* Compute SVD */
    algorithm.compute();

    dataFromStep1ForStep2[block] = algorithm.getPartialResult()->get(svd::outputOfStep1ForStep2);
    dataFromStep1ForStep3[block] = algorithm.getPartialResult()->get(svd::outputOfStep1ForStep3);
}

void computeOnMasterNode() {
    /* Create an algorithm to compute SVD on the master node */
    svd::Distributed<step2Master> algorithm;

    for (size_t i = 0; i < nBlocks; i++) {
        algorithm.input.add(svd::inputOfStep2FromStep1, i, dataFromStep1ForStep2[i]);
    }

    /* Compute SVD */
    algorithm.compute();

    svd::DistributedPartialResultPtr pres = algorithm.getPartialResult();

    for (size_t i = 0; i < nBlocks; i++) {
        dataFromStep2ForStep3[i] = pres->get(svd::outputOfStep2ForStep3, i);
    }

    svd::ResultPtr res = algorithm.getResult();

    Sigma = res->get(svd::singularValues);
    V = res->get(svd::rightSingularMatrix);
}

void finalizeComputestep1Local(size_t block) {
    /* Create an algorithm to compute SVD on the master node */
    svd::Distributed<step3Local> algorithm;

    algorithm.input.set(svd::inputOfStep3FromStep1, dataFromStep1ForStep3[block]);
    algorithm.input.set(svd::inputOfStep3FromStep2, dataFromStep2ForStep3[block]);

    /* Compute SVD */
    algorithm.compute();

    algorithm.finalizeCompute();

    svd::ResultPtr res = algorithm.getResult();

    Ui[block] = res->get(svd::leftSingularMatrix);
}
