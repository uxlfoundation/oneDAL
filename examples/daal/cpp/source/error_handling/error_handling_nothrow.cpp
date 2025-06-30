/* file: error_handling_nothrow.cpp */
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
!    C++ example of error handling mechanism without throwing exceptions
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ERROR_HANDLING_NOTHROW"></a>
 * \example error_handling_nothrow.cpp
 */

#define DAAL_NOTHROW_EXCEPTIONS

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

const std::string wrongDatasetFileName = "../data/batch/wrong.csv";

int main() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> wrongDataSource(wrongDatasetFileName);
    /* No exception was generated due to DAAL_NOTHROW_EXCEPTIONS define */

    /* Retrieve errors from FileDataSource<CSVFeatureManager> and their description. */
    std::cout << "FileDataSource expected error: " << wrongDataSource.status().getDescription()
              << std::endl;

    return 0;
}
