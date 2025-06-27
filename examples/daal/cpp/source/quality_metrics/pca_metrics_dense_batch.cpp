/* file: pca_metrics_dense_batch.cpp */
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
!    C++ example of pca quality metrics in batch processing mode.
!
!    The program computes PCA and quality
!    metrics for the PCA.
!******************************************************************************/

/**
* <a name="DAAL-EXAMPLE-CPP-PCA_QUALITY_METRIC_SET_BATCH"></a>
* \example pca_metrics_dense_batch.cpp
*/
#include "daal.h"
#include "service.h"
#include <iostream>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::pca::quality_metric;
using namespace daal::algorithms::pca::quality_metric_set;

/* Input data set parameters */
const std::string dataFileName = "../data/batch/pca_normalized.csv";
const size_t nVectors = 1000;
const size_t nComponents = 5;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &dataFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(dataFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(nVectors);

    /* Create an algorithm for principal component analysis using the SVD method */
    pca::Batch<float, pca::svdDense> algorithm;

    /* Set the algorithm input data */
    algorithm.input.set(pca::data, dataSource.getNumericTable());

    /* Compute results of the PCA algorithm */
    algorithm.compute();

    /* Create a quality metrics algorithm for explained variances, explained variances ratios and noise_variance */
    pca::quality_metric_set::Batch qms(nComponents);

    services::SharedPtr<algorithms::Input> algInput =
        qms.getInputDataCollection()->getInput(explainedVariancesMetrics);

    explained_variance::InputPtr varianceMetrics = explained_variance::Input::cast(algInput);
    varianceMetrics->set(explained_variance::eigenvalues,
                         algorithm.getResult()->get(pca::eigenvalues));

    /* Compute quality metrics of the PCA algorithm */
    qms.compute();

    /* Output quality metrics of the PCA algorithm */
    explained_variance::ResultPtr qmsResult = explained_variance::Result::cast(
        qms.getResultCollection()->getResult(explainedVariancesMetrics));
    printNumericTable(qmsResult->get(explained_variance::explainedVariances),
                      "Explained variances:");
    printNumericTable(qmsResult->get(explained_variance::explainedVariancesRatios),
                      "Explained variance ratios:");
    printNumericTable(qmsResult->get(explained_variance::noiseVariance), "Noise variance:");

    return 0;
}
