/* file: lbfgs_opt_res_dense_batch.cpp */
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
!    algorithm with optional result calculation and its usage in the next run
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LBFGS_OPT_RES_DENSE_BATCH"></a>
 * \example lbfgs_opt_res_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

const std::string datasetFileName = "../data/batch/lbfgs.csv";

const size_t nFeatures = 10;
const size_t nIterations = 1000;
const float stepLength = 1.0e-4f;

float startPoint[nFeatures + 1] = { 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };
float expectedPoint[nFeatures + 1] = { 11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

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

    services::SharedPtr<optimization_solver::mse::Batch<> > mseObjectiveFunction(
        new optimization_solver::mse::Batch<>(data->getNumberOfRows()));
    mseObjectiveFunction->input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction->input.set(optimization_solver::mse::dependentVariables,
                                    dependentVariables);

    /* Create objects to compute LBFGS result using the default method */
    optimization_solver::lbfgs::Batch<> algorithm(mseObjectiveFunction);
    algorithm.parameter.nIterations = nIterations / 2;

    algorithm.parameter.stepLengthSequence =
        NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTableIface::doAllocate, stepLength));
    algorithm.parameter.optionalResultRequired = true;

    /* Set input objects for LBFGS algorithm */
    algorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                        NumericTablePtr(new HomogenNumericTable<>(startPoint, 1, nFeatures + 1)));

    /* Compute LBFGS result */
    algorithm.compute();

    NumericTablePtr expectedCoefficients =
        NumericTablePtr(new HomogenNumericTable<>(expectedPoint, 1, nFeatures + 1));

    /* Print computed LBFGS results */
    printNumericTable(algorithm.getResult()->get(optimization_solver::iterative_solver::minimum),
                      "Resulting coefficients after first compute():");
    printNumericTable(
        algorithm.getResult()->get(optimization_solver::iterative_solver::nIterations),
        "Number of iterations performed:");

    /* Continue calculations starting from the minimal point found */
    algorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                        algorithm.getResult()->get(optimization_solver::iterative_solver::minimum));
    /* Set optional result as an optional input */
    algorithm.input.set(
        optimization_solver::iterative_solver::optionalArgument,
        algorithm.getResult()->get(optimization_solver::iterative_solver::optionalResult));

    /* Compute LBFGS result */
    algorithm.compute();

    /* Print computed LBFGS results */
    printNumericTable(expectedCoefficients, "Expected coefficients:");
    printNumericTable(algorithm.getResult()->get(optimization_solver::iterative_solver::minimum),
                      "Resulting coefficients after second compute():");
    printNumericTable(
        algorithm.getResult()->get(optimization_solver::iterative_solver::nIterations),
        "Number of iterations performed:");
    return 0;
}
