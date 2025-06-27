/* file: simple_csv_feature_modifiers.cpp */
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
 * <a name="DAAL-EXAMPLE-CPP-DATASOURCE_SIMPLE_CSV_FEATURE_MODIFIERS">
 * \example simple_csv_feature_modifiers.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal::data_management;

/* Path to the CSV to be read */
const std::string csvFileName = "../data/batch/mixed_text_and_numbers.csv";

/* Define options for CSV data source */
const CsvDataSourceOptions csvOptions = CsvDataSourceOptions::allocateNumericTable |
                                        CsvDataSourceOptions::createDictionaryFromContext |
                                        CsvDataSourceOptions::parseHeader;

/* Read CSV using default data source behavior */
void readDefault() {
    FileDataSource<CSVFeatureManager> ds(csvFileName, csvOptions);

    /* By default all numeric columns will be parsed as continuous
     * features and other columns as categorical */
    ds.loadDataBlock();

    printNumericTable(ds.getNumericTable(), "readDefault function result:");
}

/* Read CSV and do basic filtering using columns indices */
void readOnlySpecifiedColumnIndices() {
    FileDataSource<CSVFeatureManager> ds(csvFileName, csvOptions);

    /* This means that columns with indices {0, 1, 5} will be included to the output numeric
     * table and other columns will be ignored. The first argument of method 'include' specifies
     * the set of columns and the second one specifies modifier. in this case we use predefined
     * automatic modifier that automatically decides how to parse column in the best way */
    ds.getFeatureManager().addModifier(features::list(0, 1, 5), modifiers::csv::automatic());

    ds.loadDataBlock();

    printNumericTable(ds.getNumericTable(), "readOnlySpecifiedColumnIndices function result:");
}

/* Read CSV and do basic filtering using columns names */
void readOnlySpecifiedColumnNames() {
    FileDataSource<CSVFeatureManager> ds(csvFileName, csvOptions);

    /* The same as readOnlySpecifiedColumnIndices but uses column names instead of indices */
    ds.getFeatureManager().addModifier(features::list("Numeric1", "Categorical0"),
                                       modifiers::csv::automatic());

    ds.loadDataBlock();

    printNumericTable(ds.getNumericTable(), "readOnlySpecifiedColumnNames function result:");
}

/* Read CSV using multiple modifiers */
void readUsingMultipleModifiers() {
    FileDataSource<CSVFeatureManager> ds(csvFileName, csvOptions);

    ds.getFeatureManager()
        .addModifier(features::list("Numeric1"), modifiers::csv::continuous())
        .addModifier(features::list("Text1", "Categorical1"), modifiers::csv::categorical());

    ds.loadDataBlock();

    printNumericTable(ds.getNumericTable(), "readUsingMultipleModifiers function result:");
}

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &csvFileName);

    /* Read CSV using default data source behavior */
    readDefault();

    /* Read CSV and do basic filtering using columns indices */
    readOnlySpecifiedColumnIndices();

    /* Read CSV and do basic filtering using columns names */
    readOnlySpecifiedColumnNames();

    /* Read CSV using multiple modifiers */
    readUsingMultipleModifiers();

    return 0;
}
