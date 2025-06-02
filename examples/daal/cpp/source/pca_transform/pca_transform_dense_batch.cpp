/* file: pca_transform_dense_batch.cpp */
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
!    C++ example of principal component analysis transformation(PCA)
!    in the batch processing mode with reduction
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_TRANSFORM_DENSE_BATCH"></a>
 * \example pca_transform_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string dataFileName = "../data/batch/pca_transform.csv";
const size_t nVectors = 4;
const size_t nComponents = 2;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &dataFileName);
    FileDataSource<CSVFeatureManager> dataSource(dataFileName,
                                                 DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock(nVectors);

    /* Create an algorithm for principal component analysis using the SVD method */
    pca::Batch<double, pca::svdDense> pcaAlgorithm;

    pcaAlgorithm.input.set(pca::data, dataSource.getNumericTable());

    pcaAlgorithm.parameter.resultsToCompute = pca::mean | pca::variance | pca::eigenvalue;

    /* Compute results of the PCA algorithm*/
    pcaAlgorithm.compute();

    pca::ResultPtr pcaResult = pcaAlgorithm.getResult();

    /* Output basis, eigenvalues and mean values*/
    printNumericTable(pcaResult->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(pcaResult->get(pca::eigenvectors), "Eigenvectors:");

    KeyValueDataCollectionPtr resultCollection = pcaResult->get(pca::dataForTransform);
    if (resultCollection.get() != NULL) {
        NumericTablePtr eigenvaluesT = NumericTable::cast((*resultCollection)[pca::eigenvalue]);
        if (eigenvaluesT.get() != NULL)
            printNumericTable(eigenvaluesT, "Eigenvalues kv:");

        NumericTablePtr meansT = NumericTable::cast((*resultCollection)[pca::mean]);
        if (meansT.get() != NULL)
            printNumericTable(meansT, "Means kv:");

        NumericTablePtr variancesT = NumericTable::cast((*resultCollection)[pca::variance]);
        if (variancesT.get() != NULL)
            printNumericTable(variancesT, "Variances kv:");
    }

    /* Apply transform with whitening because means and eigenvalues are provided*/
    pca::transform::Batch<float> pcaTransform(nComponents);
    pcaTransform.input.set(pca::transform::data, dataSource.getNumericTable());
    pcaTransform.input.set(pca::transform::eigenvectors, pcaResult->get(pca::eigenvectors));

    pcaTransform.input.set(pca::transform::dataForTransform, pcaResult->get(pca::dataForTransform));

    pcaTransform.compute();

    /* Output transformed data */
    pca::transform::ResultPtr pcaTransformResult = pcaTransform.getResult();
    printNumericTable(pcaTransformResult->get(pca::transform::transformedData),
                      "Transformed data:");

    return 0;
}
