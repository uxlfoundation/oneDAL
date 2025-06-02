/* file: basic_statistics.cpp */
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
!    C++ example of using basic statistics
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-BASIC_STATISTICS">
 * \example basic_statistics.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;

int main() {
    std::cout << "Basic statistics example" << std::endl << std::endl;

    const size_t nObservations = 4;
    const size_t nFeatures = 4;
    float data[nFeatures * nObservations] = { 7.0f, 3.0f, 6.0f, 2.0f, 1.0f, 3.0f, 0.0f, 2.0f,
                                              9.0f, 2.0f, 6.0f, 2.0f, 3.0f, 4.0f, 7.0f, 2.0f };
    const char *csvString = "7,3,6,2\n1,3,0,2\n9,2,6,2\n3,4,7,2";

    /* Initialize StringDataSource to read data from a string in the csv format */
    StringDataSource<CSVFeatureManager> dataSource((daal::byte *)csvString,
                                                   DataSource::doAllocateNumericTable,
                                                   DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();
    NumericTablePtr table = dataSource.getNumericTable();

    /* Get basic statistics from the table. They were calculated inside DataSource for each column. */
    NumericTablePtr min = table->basicStatistics.get(NumericTableIface::minimum);
    NumericTablePtr max = table->basicStatistics.get(NumericTableIface::maximum);
    NumericTablePtr sum = table->basicStatistics.get(NumericTableIface::sum);
    NumericTablePtr sumSquares = table->basicStatistics.get(NumericTableIface::sumSquares);

    /* Print calculated basic statistics */
    printNumericTable(table.get(), "Basic statistics of table:");
    printNumericTable(min.get(), "Minimum:");
    printNumericTable(max.get(), "Maximum:");
    printNumericTable(sum.get(), "Sum:");
    printNumericTable(sumSquares.get(), "SumSquares:");

    /* Create NumericTable with the same data. But in this case basic statistics are not calculated. */
    SharedPtr<HomogenNumericTable<> > dataTable =
        HomogenNumericTable<>::create(data, nFeatures, nObservations);
    checkPtr(dataTable.get());

    /* Set basic statistics in the new NumericTable */
    dataTable->basicStatistics.set(NumericTableIface::minimum, min);
    dataTable->basicStatistics.set(NumericTableIface::maximum, max);
    dataTable->basicStatistics.set(NumericTableIface::sum, sum);
    dataTable->basicStatistics.set(NumericTableIface::sumSquares, sumSquares);

    /* Print basic statistics those were set */
    printNumericTable(dataTable.get(), "New table:");
    printNumericTable(dataTable->basicStatistics.get(NumericTableIface::minimum).get(), "Minimum:");
    printNumericTable(dataTable->basicStatistics.get(NumericTableIface::maximum).get(), "Maximum:");
    printNumericTable(dataTable->basicStatistics.get(NumericTableIface::sum).get(), "Sum:");
    printNumericTable(dataTable->basicStatistics.get(NumericTableIface::sumSquares).get(),
                      "SumSquares:");

    return 0;
}
