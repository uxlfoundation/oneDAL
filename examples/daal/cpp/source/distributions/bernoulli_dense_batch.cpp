/* file: bernoulli_dense_batch.cpp */
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
!    C++ example of bernoulli distribution usage
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-BERNOULLI_DENSE_BATCH"></a>
 * \example bernoulli_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::engines;
using namespace daal::algorithms::distributions;

int main() {
    /* Create input table to fill with random numbers */
    NumericTablePtr dataTable(
        new HomogenNumericTable<int>(DictionaryIface::equal, 1, 10, NumericTable::doAllocate));

    /* Create the algorithm */
    bernoulli::Batch<> bernoulli(0.5);

    /* Set the algorithm input */
    bernoulli.input.set(distributions::tableToFill, dataTable);

    /* Set the Mersenne Twister engine to the distribution */
    bernoulli.parameter.engine = mt19937::Batch<>::create(777);

    /* Perform computations */
    bernoulli.compute();

    /* Print the results */
    printNumericTable(dataTable, "Bernoulli distribution output:");

    return 0;
}
