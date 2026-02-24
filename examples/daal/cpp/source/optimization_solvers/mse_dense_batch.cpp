/* file: mse_dense_batch.cpp */
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
!    C++ example of the mean squared error objective function
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-MSE_BATCH"></a>
 * \example mse_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

const std::string datasetFileName = "../data/batch/mse.csv";
const size_t nFeatures = 3;

float argumentValue[nFeatures + 1] = { -1, 0.1f, 0.15f, -0.5f };

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::notAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for data and values for dependent variable */
    NumericTablePtr data(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr dependentVariables(
        new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(mergedData.get());

    size_t nVectors = data->getNumberOfRows();

    /* Create the MSE objective function objects to compute the MSE objective function result using the default method */
    optimization_solver::mse::Batch<> mseObjectiveFunction(nVectors);

    /* Set input objects for the MSE objective function */
    mseObjectiveFunction.input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction.input.set(optimization_solver::mse::dependentVariables,
                                   dependentVariables);
    mseObjectiveFunction.input.set(
        optimization_solver::mse::argument,
        NumericTablePtr(new HomogenNumericTable<>(argumentValue, 1, nFeatures + 1)));
    mseObjectiveFunction.parameter().resultsToCompute =
        optimization_solver::objective_function::gradient |
        optimization_solver::objective_function::value |
        optimization_solver::objective_function::hessian;

    /* Compute the MSE objective function result */
    mseObjectiveFunction.compute();

    /* Print computed the MSE objective function result */
    printNumericTable(
        mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::valueIdx),
        "Value");
    printNumericTable(
        mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::gradientIdx),
        "Gradient");
    printNumericTable(
        mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::hessianIdx),
        "Hessian");

    return 0;
}
