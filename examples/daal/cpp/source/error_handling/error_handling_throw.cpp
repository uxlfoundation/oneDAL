/* file: error_handling_throw.cpp */
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
 * <a name="DAAL-EXAMPLE-CPP-ERROR_HANDLING_THROW"></a>
 * \example error_handling_throw.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

const std::string wrongDatasetFileName = "../data/batch/wrong.csv";

int main() {
    try {
        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> wrongDataSource(wrongDatasetFileName);
        /* An exception was generated due to absense DAAL_NOTHROW_EXCEPTIONS define by default */
    }
    catch (daal::services::Exception& e) {
        /* Retrieve the description of the generated exception. */
        std::cout << "FileDataSource expected error: " << e.what() << std::endl;
    }

    return 0;
}
