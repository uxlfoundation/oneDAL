/* file: kernel_func_lin_csr_batch.cpp */
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
!    C++ example of computing a linear kernel function in the batch processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-KERNEL_FUNCTION_LINEAR_CSR_BATCH"></a>
 * \example kernel_func_lin_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string leftDatasetFileName = "../data/batch/kernel_function_csr.csv";
const std::string rightDatasetFileName = "../data/batch/kernel_function_csr.csv";

/* Kernel algorithm parameters */
const double k = 1.0; /* Linear kernel coefficient in the k(X,Y) + b model */
const double b = 0.0; /* Linear kernel coefficient in the k(X,Y) + b model */

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &leftDatasetFileName);
    checkArguments(argc, argv, 1, &rightDatasetFileName);

    /* Read datasetFileName from a file and create a numeric tables to store input data */
    CSRNumericTablePtr leftData(createSparseTable<float>(leftDatasetFileName));
    CSRNumericTablePtr rightData(createSparseTable<float>(rightDatasetFileName));

    /* Create algorithm objects for the kernel algorithm using the default method */
    kernel_function::linear::Batch<float, kernel_function::linear::fastCSR> algorithm;

    /* Set the kernel algorithm parameter */
    algorithm.parameter.k = k;
    algorithm.parameter.b = b;
    algorithm.parameter.computationMode = kernel_function::matrixMatrix;

    /* Set an input data table for the algorithm */
    algorithm.input.set(kernel_function::X, leftData);
    algorithm.input.set(kernel_function::Y, rightData);

    /* Compute the linear kernel function */
    algorithm.compute();

    /* Get the computed results */
    kernel_function::ResultPtr result = algorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(kernel_function::values), "Values");

    return 0;
}
