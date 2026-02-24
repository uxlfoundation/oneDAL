/* file: lbfgs_cr_entr_loss_dense_batch.cpp */
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
!    C++ example of the limited memory Broyden-Fletcher-Goldfarb-Shanno
!    algorithm with cross entropy loss function
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LBFGS_CR_ENTR_LOSS-BATCH"></a>
 * \example lbfgs_cr_entr_loss_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

const std::string datasetFileName = "../data/batch/logreg_train.csv";
const size_t nFeatures = 6; /* Number of features in training and testing data sets */
const size_t nClasses = 5; /* Number of classes */
const size_t nIterations = 1000;
const float stepLength = 1.0e-4f;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::notAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for input data and dependent variables */
    NumericTablePtr data(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr dependentVariables(
        new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from input file */
    dataSource.loadDataBlock(mergedData.get());

    services::SharedPtr<optimization_solver::cross_entropy_loss::Batch<> > func =
        optimization_solver::cross_entropy_loss::Batch<>::create(nClasses, data->getNumberOfRows());
    func->input.set(optimization_solver::cross_entropy_loss::data, data);
    func->input.set(optimization_solver::cross_entropy_loss::dependentVariables,
                    dependentVariables);

    /* Create objects to compute LBFGS result using the default method */
    optimization_solver::lbfgs::Batch<> algorithm(func);
    algorithm.parameter.nIterations = nIterations;
    algorithm.parameter.stepLengthSequence =
        NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTableIface::doAllocate, stepLength));

    const size_t nParameters = nClasses * (nFeatures + 1);
    DAAL_DATA_TYPE initialPoint[nParameters];
    for (size_t i = 0; i < nParameters; ++i)
        initialPoint[i] = 0.001f;

    /* Set input objects for LBFGS algorithm */
    algorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                        HomogenNumericTable<>::create(initialPoint, 1, nParameters));

    /* Compute LBFGS result */
    algorithm.compute();

    DAAL_DATA_TYPE expectedPoint[nParameters] = {
        -2.277f,  2.836f, 14.985f, 0.511f,   7.510f,  -2.831f, -5.814f, -0.033f, 13.227f,
        -24.447f, 3.730f, 10.394f, -10.461f, -0.766f, 0.077f,  1.558f,  -1.133f, 2.884f,
        -3.825f,  7.699f, 2.421f,  -0.135f,  -6.996f, 1.785f,  -2.294f, -9.819f, 1.692f,
        -0.725f,  0.069f, -8.41f,  1.458f,   -3.306f, -4.719f, 5.507f,  -1.642f
    };

    NumericTablePtr expectedCoefficients =
        HomogenNumericTable<>::create(expectedPoint, 1, nParameters);

    /* Print computed LBFGS results */
    printNumericTable(expectedCoefficients, "Expected coefficients:");
    printNumericTable(algorithm.getResult()->get(optimization_solver::iterative_solver::minimum),
                      "Resulting coefficients:");
    printNumericTable(
        algorithm.getResult()->get(optimization_solver::iterative_solver::nIterations),
        "Number of iterations performed:");
    return 0;
}
