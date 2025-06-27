/* file: custom_csv_feature_modifiers.cpp */
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
!    C++ example of modifiers usage with file data source
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASOURCE_CUSTOM_CSV_FEATURE_MODIFIERS">
 * \example custom_csv_feature_modifiers.cpp
 */

#include <cassert>
#include <algorithm>

#include "daal.h"
#include "service.h"

using namespace daal::data_management;

/** User-defined feature modifier that computes a square for every feature */
class MySquaringModifier : public modifiers::csv::FeatureModifier {
public:
    /* This method is called for every row in CSV file */
    virtual void apply(modifiers::csv::Context& context) {
        const size_t numberOfTokens = context.getNumberOfTokens();
        daal::services::BufferView<DAAL_DATA_TYPE> outputBuffer = context.getOutputBuffer();

        /* By default number of tokens (token is one word separated by commas) is equals to the
         * buffer size. This behavior can be redefined by calling 'setNumberOfOutputFeatures' on
         * initialization stage of the modifier (see 'MyMaxFeatureModifier') */
        assert(numberOfTokens == outputBuffer.size());

        for (size_t i = 0; i < numberOfTokens; i++) {
            const float x = context.getTokenAs<float>(i);
            outputBuffer[i] = x * x;
        }
    }
};

/** User-defined feature modifier that selects max element among all features  */
class MyMaxFeatureModifier : public modifiers::csv::FeatureModifier {
public:
    /* This method is called once before CSV parsing */
    virtual void initialize(modifiers::csv::Config& config) {
        /* Set number of output features for the modifier. We assume modifier
         * computes function y = max { x_1, ..., x_n }, where x_i is input
         * features and y is output feature, so there is single output feature  */
        config.setNumberOfOutputFeatures(1);
    }

    /* This method is called for every row in CSV file */
    virtual void apply(modifiers::csv::Context& context) {
        const size_t numberOfTokens = context.getNumberOfTokens();

        /* Iterate throughout tokens, parse every token as float and compute max value  */
        float maxFeature = context.getTokenAs<float>(0);
        for (size_t i = 1; i < numberOfTokens; i++) {
            maxFeature = std::max(maxFeature, context.getTokenAs<float>(i));
        }

        /* Write max value to the output buffer, buffer size is equal to the
         * number of output features that specified in 'initialize' method */
        context.getOutputBuffer()[0] = maxFeature;
    }
};

int main(int argc, char* argv[]) {
    /* Path to the CSV to be read */
    const std::string csvFileName = "../data/batch/mixed_text_and_numbers.csv";

    checkArguments(argc, argv, 1, &csvFileName);

    /* Define options for CSV data source */
    const CsvDataSourceOptions csvOptions = CsvDataSourceOptions::allocateNumericTable |
                                            CsvDataSourceOptions::createDictionaryFromContext |
                                            CsvDataSourceOptions::parseHeader;

    /* Define CSV file data source */
    FileDataSource<CSVFeatureManager> ds(csvFileName, csvOptions);

    /* Configure format of output numeric table by applying modifiers.
     * Output numeric table will have the following format:
     * | Numeric1 | Numeric2 ^ 2 | Numeric5 ^ 2 | max(Numeric0, Numeric5) | */
    ds.getFeatureManager()
        .addModifier(features::list("Numeric1"), modifiers::csv::continuous())
        .addModifier(features::list("Numeric2", "Numeric5"),
                     modifiers::csv::custom<MySquaringModifier>())
        .addModifier(features::list("Numeric0", "Numeric5"),
                     modifiers::csv::custom<MyMaxFeatureModifier>());

    /* Load and parse CSV file */
    ds.loadDataBlock();

    printNumericTable(ds.getNumericTable(), "Loaded numeric table:");

    return 0;
}
