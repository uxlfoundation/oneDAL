/* file: service.h */
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
!    Auxiliary functions used in C++ examples
!******************************************************************************/

#ifndef _SERVICE_H
#define _SERVICE_H

#include "daal.h"

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdarg>
#include <vector>
#include <queue>

#include "error_handling.h"

size_t readTextFile(const std::string &datasetFileName, daal::byte **data) {
    std::ifstream file(datasetFileName.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fileOpenError(datasetFileName.c_str());
    }

    std::streampos end = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t fileSize = static_cast<size_t>(end);

    (*data) = new daal::byte[fileSize];
    checkAllocation(data);

    if (!file.read((char *)(*data), fileSize)) {
        delete[] data;
        fileReadError();
    }

    return fileSize;
}

template <typename item_type>
void readRowUnknownLength(char *line, std::vector<item_type> &data) {
    size_t n = 0;
    const char *prevDelim = line - 1;
    char *ptr = line;
    for (; *ptr; ++ptr) {
        if (*ptr == ',' || *ptr == '\r') {
            if (prevDelim != ptr - 1)
                ++n;
            *ptr = ' ';
            prevDelim = ptr;
        }
    }
    if (prevDelim != ptr - 1)
        ++n;
    data.resize(n);
    std::stringstream iss(line);
    for (size_t i = 0; i < n; ++i) {
        iss >> data[i];
    }
}

template <typename item_type>
daal::data_management::CSRNumericTable *createSparseTable(const std::string &datasetFileName) {
    std::ifstream file(datasetFileName.c_str());

    if (!file.is_open()) {
        fileOpenError(datasetFileName.c_str());
    }

    std::string str;

    //read row offsets
    std::getline(file, str);
    std::vector<size_t> rowOffsets;
    readRowUnknownLength<size_t>(&str[0], rowOffsets);
    if (!rowOffsets.size())
        return NULL;
    const size_t nVectors = rowOffsets.size() - 1;

    //read cols indices
    std::getline(file, str);
    std::vector<size_t> colIndices;
    readRowUnknownLength<size_t>(&str[0], colIndices);

    //read values
    std::getline(file, str);
    std::vector<item_type> data;
    readRowUnknownLength<item_type>(&str[0], data);
    const size_t nNonZeros = data.size();

    size_t maxCol = 0;
    for (size_t i = 0; i < colIndices.size(); ++i) {
        if (colIndices[i] > maxCol)
            maxCol = colIndices[i];
    }
    const size_t nFeatures = maxCol;

    if (!nFeatures || !nVectors || colIndices.size() != nNonZeros ||
        nNonZeros != (rowOffsets[nVectors] - 1)) {
        sparceFileReadError();
    }

    size_t *resultRowOffsets = NULL;
    size_t *resultColIndices = NULL;
    item_type *resultData = NULL;
    daal::data_management::CSRNumericTable *numericTable =
        new daal::data_management::CSRNumericTable(resultData,
                                                   resultColIndices,
                                                   resultRowOffsets,
                                                   nFeatures,
                                                   nVectors);
    numericTable->allocateDataMemory(nNonZeros);
    numericTable->getArrays<item_type>(&resultData, &resultColIndices, &resultRowOffsets);
    for (size_t i = 0; i < nNonZeros; ++i) {
        resultData[i] = data[i];
        resultColIndices[i] = colIndices[i];
    }
    for (size_t i = 0; i < nVectors + 1; ++i) {
        resultRowOffsets[i] = rowOffsets[i];
    }
    return numericTable;
}

void printAprioriItemsets(daal::data_management::NumericTablePtr largeItemsetsTable,
                          daal::data_management::NumericTablePtr largeItemsetsSupportTable,
                          size_t nItemsetToPrint = 20) {
    using namespace daal::data_management;

    size_t largeItemsetCount = largeItemsetsSupportTable->getNumberOfRows();
    size_t nItemsInLargeItemsets = largeItemsetsTable->getNumberOfRows();

    BlockDescriptor<int> block1;
    largeItemsetsTable->getBlockOfRows(0, nItemsInLargeItemsets, readOnly, block1);
    int *largeItemsets = block1.getBlockPtr();

    BlockDescriptor<int> block2;
    largeItemsetsSupportTable->getBlockOfRows(0, largeItemsetCount, readOnly, block2);
    int *largeItemsetsSupportData = block2.getBlockPtr();

    std::vector<std::vector<size_t> > largeItemsetsVector;
    largeItemsetsVector.resize(largeItemsetCount);

    for (size_t i = 0; i < nItemsInLargeItemsets; i++) {
        largeItemsetsVector[largeItemsets[2 * i]].push_back(largeItemsets[2 * i + 1]);
    }

    std::vector<size_t> supportVector;
    supportVector.resize(largeItemsetCount);

    for (size_t i = 0; i < largeItemsetCount; i++) {
        supportVector[largeItemsetsSupportData[2 * i]] = largeItemsetsSupportData[2 * i + 1];
    }

    std::cout << std::endl << "Apriori example program results" << std::endl;

    std::cout << std::endl << "Last " << nItemsetToPrint << " large itemsets: " << std::endl;
    std::cout << std::endl
              << "Itemset"
              << "\t\t\tSupport" << std::endl;

    size_t iMin = (((largeItemsetCount > nItemsetToPrint) && (nItemsetToPrint != 0))
                       ? largeItemsetCount - nItemsetToPrint
                       : 0);
    for (size_t i = iMin; i < largeItemsetCount; i++) {
        std::cout << "{";
        for (size_t l = 0; l < largeItemsetsVector[i].size() - 1; l++) {
            std::cout << largeItemsetsVector[i][l] << ", ";
        }
        std::cout << largeItemsetsVector[i][largeItemsetsVector[i].size() - 1] << "}\t\t";

        std::cout << supportVector[i] << std::endl;
    }

    largeItemsetsTable->releaseBlockOfRows(block1);
    largeItemsetsSupportTable->releaseBlockOfRows(block2);
}

void printAprioriRules(daal::data_management::NumericTablePtr leftItemsTable,
                       daal::data_management::NumericTablePtr rightItemsTable,
                       daal::data_management::NumericTablePtr confidenceTable,
                       size_t nRulesToPrint = 20) {
    using namespace daal::data_management;

    size_t nRules = confidenceTable->getNumberOfRows();
    size_t nLeftItems = leftItemsTable->getNumberOfRows();
    size_t nRightItems = rightItemsTable->getNumberOfRows();

    BlockDescriptor<int> block1;
    leftItemsTable->getBlockOfRows(0, nLeftItems, readOnly, block1);
    int *leftItems = block1.getBlockPtr();

    BlockDescriptor<int> block2;
    rightItemsTable->getBlockOfRows(0, nRightItems, readOnly, block2);
    int *rightItems = block2.getBlockPtr();

    BlockDescriptor<DAAL_DATA_TYPE> block3;
    confidenceTable->getBlockOfRows(0, nRules, readOnly, block3);
    DAAL_DATA_TYPE *confidence = block3.getBlockPtr();

    std::vector<std::vector<size_t> > leftItemsVector;
    leftItemsVector.resize(nRules);

    if (nRules == 0) {
        std::cout << std::endl << "No association rules were found " << std::endl;
        return;
    }

    for (size_t i = 0; i < nLeftItems; i++) {
        leftItemsVector[leftItems[2 * i]].push_back(leftItems[2 * i + 1]);
    }

    std::vector<std::vector<size_t> > rightItemsVector;
    rightItemsVector.resize(nRules);

    for (size_t i = 0; i < nRightItems; i++) {
        rightItemsVector[rightItems[2 * i]].push_back(rightItems[2 * i + 1]);
    }

    std::vector<DAAL_DATA_TYPE> confidenceVector;
    confidenceVector.resize(nRules);

    for (size_t i = 0; i < nRules; i++) {
        confidenceVector[i] = confidence[i];
    }

    std::cout << std::endl << "Last " << nRulesToPrint << " association rules: " << std::endl;
    std::cout << std::endl
              << "Rule"
              << "\t\t\t\tConfidence" << std::endl;
    size_t iMin =
        (((nRules > nRulesToPrint) && (nRulesToPrint != 0)) ? (nRules - nRulesToPrint) : 0);

    for (size_t i = iMin; i < nRules; i++) {
        std::cout << "{";
        for (size_t l = 0; l < leftItemsVector[i].size() - 1; l++) {
            std::cout << leftItemsVector[i][l] << ", ";
        }
        std::cout << leftItemsVector[i][leftItemsVector[i].size() - 1] << "} => {";

        for (size_t l = 0; l < rightItemsVector[i].size() - 1; l++) {
            std::cout << rightItemsVector[i][l] << ", ";
        }
        std::cout << rightItemsVector[i][rightItemsVector[i].size() - 1] << "}\t\t";

        std::cout << confidenceVector[i] << std::endl;
    }

    leftItemsTable->releaseBlockOfRows(block1);
    rightItemsTable->releaseBlockOfRows(block2);
    confidenceTable->releaseBlockOfRows(block3);
}

bool isFull(daal::data_management::NumericTableIface::StorageLayout layout) {
    int layoutInt = (int)layout;
    if (daal::data_management::packed_mask & layoutInt) {
        return false;
    }
    return true;
}

bool isUpper(daal::data_management::NumericTableIface::StorageLayout layout) {
    using daal::data_management::NumericTableIface;

    if (layout == NumericTableIface::upperPackedSymmetricMatrix ||
        layout == NumericTableIface::upperPackedTriangularMatrix) {
        return true;
    }
    return false;
}

bool isLower(daal::data_management::NumericTableIface::StorageLayout layout) {
    using daal::data_management::NumericTableIface;

    if (layout == NumericTableIface::lowerPackedSymmetricMatrix ||
        layout == NumericTableIface::lowerPackedTriangularMatrix) {
        return true;
    }
    return false;
}

template <typename T>
void printArray(T *array,
                const size_t nPrintedCols,
                const size_t nPrintedRows,
                const size_t nCols,
                std::string message,
                size_t interval = 10) {
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    for (size_t i = 0; i < nPrintedRows; i++) {
        for (size_t j = 0; j < nPrintedCols; j++) {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                      << std::setprecision(3);
            std::cout << array[i * nCols + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void printArray(T *array,
                const size_t nCols,
                const size_t nRows,
                std::string message,
                size_t interval = 10) {
    printArray(array, nCols, nRows, nCols, message, interval);
}

template <typename T>
void printLowerArray(T *array,
                     const size_t nPrintedRows,
                     std::string message,
                     size_t interval = 10) {
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    int ind = 0;
    for (size_t i = 0; i < nPrintedRows; i++) {
        for (size_t j = 0; j <= i; j++) {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                      << std::setprecision(3);
            std::cout << array[ind++];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void printUpperArray(T *array,
                     const size_t nPrintedCols,
                     const size_t nPrintedRows,
                     const size_t nCols,
                     std::string message,
                     size_t interval = 10) {
    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    int ind = 0;
    for (size_t i = 0; i < nPrintedRows; i++) {
        for (size_t j = 0; j < i; j++) {
            std::cout << "          ";
        }
        for (size_t j = i; j < nPrintedCols; j++) {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                      << std::setprecision(3);
            std::cout << array[ind++];
        }
        for (size_t j = nPrintedCols; j < nCols; j++) {
            ind++;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printNumericTable(daal::data_management::NumericTable *dataTable,
                       const char *message = "",
                       size_t nPrintedRows = 0,
                       size_t nPrintedCols = 0,
                       size_t interval = 10) {
    using namespace daal::data_management;

    size_t nRows = dataTable->getNumberOfRows();
    size_t nCols = dataTable->getNumberOfColumns();
    NumericTableIface::StorageLayout layout = dataTable->getDataLayout();

    if (nPrintedRows != 0) {
        nPrintedRows = std::min(nRows, nPrintedRows);
    }
    else {
        nPrintedRows = nRows;
    }

    if (nPrintedCols != 0) {
        nPrintedCols = std::min(nCols, nPrintedCols);
    }
    else {
        nPrintedCols = nCols;
    }

    BlockDescriptor<DAAL_DATA_TYPE> block;
    if (isFull(layout) || layout == NumericTableIface::csrArray) {
        dataTable->getBlockOfRows(0, nRows, readOnly, block);
        printArray<DAAL_DATA_TYPE>(block.getBlockPtr(),
                                   nPrintedCols,
                                   nPrintedRows,
                                   nCols,
                                   message,
                                   interval);
        dataTable->releaseBlockOfRows(block);
    }
    else {
        PackedArrayNumericTableIface *packedTable =
            dynamic_cast<PackedArrayNumericTableIface *>(dataTable);
        packedTable->getPackedArray(readOnly, block);
        if (isLower(layout)) {
            printLowerArray<DAAL_DATA_TYPE>(block.getBlockPtr(), nPrintedRows, message, interval);
        }
        else if (isUpper(layout)) {
            printUpperArray<DAAL_DATA_TYPE>(block.getBlockPtr(),
                                            nPrintedCols,
                                            nPrintedRows,
                                            nCols,
                                            message,
                                            interval);
        }
        packedTable->releasePackedArray(block);
    }
}

void printNumericTable(daal::data_management::NumericTable &dataTable,
                       const char *message = "",
                       size_t nPrintedRows = 0,
                       size_t nPrintedCols = 0,
                       size_t interval = 10) {
    printNumericTable(&dataTable, message, nPrintedRows, nPrintedCols, interval);
}

void printNumericTable(const daal::data_management::NumericTablePtr &dataTable,
                       const char *message = "",
                       size_t nPrintedRows = 0,
                       size_t nPrintedCols = 0,
                       size_t interval = 10) {
    printNumericTable(dataTable.get(), message, nPrintedRows, nPrintedCols, interval);
}

void printPackedNumericTable(daal::data_management::NumericTable *dataTable,
                             size_t nFeatures,
                             const char *message = "",
                             size_t interval = 10) {
    using namespace daal::data_management;

    BlockDescriptor<DAAL_DATA_TYPE> block;

    dataTable->getBlockOfRows(0, 1, readOnly, block);

    DAAL_DATA_TYPE *data = block.getBlockPtr();

    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    size_t index = 0;
    for (size_t i = 0; i < nFeatures; i++) {
        for (size_t j = 0; j <= i; j++, index++) {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                      << std::setprecision(3);
            std::cout << data[index];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dataTable->releaseBlockOfRows(block);
}

void printPackedNumericTable(daal::data_management::NumericTable &dataTable,
                             size_t nFeatures,
                             const char *message = "",
                             size_t interval = 10) {
    printPackedNumericTable(&dataTable, nFeatures, message, interval);
}

/**
 * @brief Class for defining printing precision
 * @tparam type type name to define presicion value
 */
template <typename type>
struct PrecisionValue {
    static int get() {
        return 0;
    }
};

/**
 * @brief Partial specialization for float type
 * @return Pronting precision for float type
 */
template <>
struct PrecisionValue<float> {
    static int get() {
        return 3;
    }
};

/**
 * @brief Partial specialization for double type
 * @return Pronting precision for double type
 */
template <>
struct PrecisionValue<double> {
    static int get() {
        return 3;
    }
};

template <typename type1, typename type2>
void printNumericTables(daal::data_management::NumericTable *dataTable1,
                        daal::data_management::NumericTable *dataTable2,
                        const char *title1 = "",
                        const char *title2 = "",
                        const char *message = "",
                        size_t nPrintedRows = 0,
                        size_t interval = 15) {
    using namespace daal::data_management;

    size_t nRows1 = dataTable1->getNumberOfRows();
    size_t nRows2 = dataTable2->getNumberOfRows();
    size_t nCols1 = dataTable1->getNumberOfColumns();
    size_t nCols2 = dataTable2->getNumberOfColumns();

    BlockDescriptor<type1> block1;
    BlockDescriptor<type2> block2;

    size_t nRows = std::min(nRows1, nRows2);
    if (nPrintedRows != 0) {
        nRows = std::min(std::min(nRows1, nRows2), nPrintedRows);
    }

    dataTable1->getBlockOfRows(0, nRows, readOnly, block1);
    dataTable2->getBlockOfRows(0, nRows, readOnly, block2);

    type1 *data1 = block1.getBlockPtr();
    type2 *data2 = block2.getBlockPtr();

    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    std::cout << std::setw(interval * nCols1) << title1;
    std::cout << std::setw(interval * nCols2) << title2 << std::endl;
    for (size_t i = 0; i < nRows; i++) {
        for (size_t j = 0; j < nCols1; j++) {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                      << std::setprecision(PrecisionValue<type1>::get());
            std::cout << data1[i * nCols1 + j];
        }
        for (size_t j = 0; j < nCols2; j++) {
            std::cout << std::setprecision(PrecisionValue<type2>::get()) << std::setw(interval)
                      << data2[i * nCols2 + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dataTable1->releaseBlockOfRows(block1);
    dataTable2->releaseBlockOfRows(block2);
}

template <typename type1, typename type2>
void printNumericTables(daal::data_management::NumericTable *dataTable1,
                        daal::data_management::NumericTable &dataTable2,
                        const char *title1 = "",
                        const char *title2 = "",
                        const char *message = "",
                        size_t nPrintedRows = 0,
                        size_t interval = 15) {
    printNumericTables<type1, type2>(dataTable1,
                                     &dataTable2,
                                     title1,
                                     title2,
                                     message,
                                     nPrintedRows,
                                     interval);
}

void printNumericTables(daal::data_management::NumericTable *dataTable1,
                        daal::data_management::NumericTable *dataTable2,
                        const char *title1 = "",
                        const char *title2 = "",
                        const char *message = "",
                        size_t nPrintedRows = 0,
                        size_t interval = 15) {
    using namespace daal::data_management;

    size_t nRows1 = dataTable1->getNumberOfRows();
    size_t nRows2 = dataTable2->getNumberOfRows();
    size_t nCols1 = dataTable1->getNumberOfColumns();
    size_t nCols2 = dataTable2->getNumberOfColumns();

    BlockDescriptor<DAAL_DATA_TYPE> block1;
    BlockDescriptor<DAAL_DATA_TYPE> block2;

    size_t nRows = std::min(nRows1, nRows2);
    if (nPrintedRows != 0) {
        nRows = std::min(std::min(nRows1, nRows2), nPrintedRows);
    }

    dataTable1->getBlockOfRows(0, nRows, readOnly, block1);
    dataTable2->getBlockOfRows(0, nRows, readOnly, block2);

    DAAL_DATA_TYPE *data1 = block1.getBlockPtr();
    DAAL_DATA_TYPE *data2 = block2.getBlockPtr();

    std::cout << std::setiosflags(std::ios::left);
    std::cout << message << std::endl;
    std::cout << std::setw(interval * nCols1) << title1;
    std::cout << std::setw(interval * nCols2) << title2 << std::endl;
    for (size_t i = 0; i < nRows; i++) {
        for (size_t j = 0; j < nCols1; j++) {
            std::cout << std::setw(interval) << std::setiosflags(std::ios::fixed)
                      << std::setprecision(3);
            std::cout << data1[i * nCols1 + j];
        }
        for (size_t j = 0; j < nCols2; j++) {
            std::cout << std::setprecision(0) << std::setw(interval) << data2[i * nCols2 + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    dataTable1->releaseBlockOfRows(block1);
    dataTable2->releaseBlockOfRows(block2);
}

void printNumericTables(daal::data_management::NumericTable *dataTable1,
                        daal::data_management::NumericTable &dataTable2,
                        const char *title1 = "",
                        const char *title2 = "",
                        const char *message = "",
                        size_t nPrintedRows = 0,
                        size_t interval = 15) {
    printNumericTables(dataTable1, &dataTable2, title1, title2, message, nPrintedRows, interval);
}

template <typename type1, typename type2>
void printNumericTables(daal::data_management::NumericTablePtr dataTable1,
                        daal::data_management::NumericTablePtr dataTable2,
                        const char *title1 = "",
                        const char *title2 = "",
                        const char *message = "",
                        size_t nPrintedRows = 0,
                        size_t interval = 15) {
    printNumericTables<type1, type2>(dataTable1.get(),
                                     dataTable2.get(),
                                     title1,
                                     title2,
                                     message,
                                     nPrintedRows,
                                     interval);
}

// The function tries to find the file `name` in several possible directories.
// This is useful because CMake and Bazel may run the program from different working directories,
// so relative paths to data files can differ.
inline const std::string get_data_path(const std::string &name) {
    const std::vector<std::string> paths = { "../data", "examples/daal/data" };
    for (const auto &path : paths) {
        const std::string try_path = path + "/" + name;
        if (std::ifstream{ try_path }.good()) {
            return try_path;
        }
    }

    return name;
}

void checkArguments(int argc, char *argv[], int count, ...) {
    std::string **const filelist = new std::string *[count];

    va_list ap;
    va_start(ap, count);
    for (int i = 0; i < count; i++) {
        filelist[i] = va_arg(ap, std::string *);
    }
    va_end(ap);

    if (argc == 1) {
        for (int i = 0; i < count; i++) {
            *(filelist[i]) = get_data_path(*(filelist[i]));
        }
    }
    else if (argc == count + 1) {
        bool all_exist = true;
        for (int i = 0; i < count; i++) {
            if (!std::ifstream{ argv[i + 1] }.good()) {
                all_exist = false;
                break;
            }
        }

        if (all_exist) {
            for (int i = 0; i < count; i++) {
                *(filelist[i]) = argv[i + 1];
            }
        }
        else {
            std::cout << "Warning: Try to open default datasetFileNames" << std::endl;
            for (int i = 0; i < count; i++) {
                *(filelist[i]) = get_data_path(*(filelist[i]));
            }
        }
    }
    else {
        std::cout << "Usage: " << argv[0] << " [ ";
        for (int i = 0; i < count; i++) {
            std::cout << "<filename_" << i << "> ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Warning: Try to open default datasetFileNames" << std::endl;
        for (int i = 0; i < count; i++) {
            *(filelist[i]) = get_data_path(*(filelist[i]));
        }
    }

    delete[] filelist;
}

void copyBytes(daal::byte *dst, daal::byte *src, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

size_t checkBytes(daal::byte *dst, daal::byte *src, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (dst[i] != src[i]) {
            return i + 1;
        }
    }
    return 0;
}

static const unsigned int crcRem[] = {
    0x00000000, 0x741B8CD6, 0xE83719AC, 0x9C2C957A, 0xA475BF8E, 0xD06E3358, 0x4C42A622, 0x38592AF4,
    0x3CF0F3CA, 0x48EB7F1C, 0xD4C7EA66, 0xA0DC66B0, 0x98854C44, 0xEC9EC092, 0x70B255E8, 0x04A9D93E,
    0x79E1E794, 0x0DFA6B42, 0x91D6FE38, 0xE5CD72EE, 0xDD94581A, 0xA98FD4CC, 0x35A341B6, 0x41B8CD60,
    0x4511145E, 0x310A9888, 0xAD260DF2, 0xD93D8124, 0xE164ABD0, 0x957F2706, 0x0953B27C, 0x7D483EAA,
    0xF3C3CF28, 0x87D843FE, 0x1BF4D684, 0x6FEF5A52, 0x57B670A6, 0x23ADFC70, 0xBF81690A, 0xCB9AE5DC,
    0xCF333CE2, 0xBB28B034, 0x2704254E, 0x531FA998, 0x6B46836C, 0x1F5D0FBA, 0x83719AC0, 0xF76A1616,
    0x8A2228BC, 0xFE39A46A, 0x62153110, 0x160EBDC6, 0x2E579732, 0x5A4C1BE4, 0xC6608E9E, 0xB27B0248,
    0xB6D2DB76, 0xC2C957A0, 0x5EE5C2DA, 0x2AFE4E0C, 0x12A764F8, 0x66BCE82E, 0xFA907D54, 0x8E8BF182,
    0x939C1286, 0xE7879E50, 0x7BAB0B2A, 0x0FB087FC, 0x37E9AD08, 0x43F221DE, 0xDFDEB4A4, 0xABC53872,
    0xAF6CE14C, 0xDB776D9A, 0x475BF8E0, 0x33407436, 0x0B195EC2, 0x7F02D214, 0xE32E476E, 0x9735CBB8,
    0xEA7DF512, 0x9E6679C4, 0x024AECBE, 0x76516068, 0x4E084A9C, 0x3A13C64A, 0xA63F5330, 0xD224DFE6,
    0xD68D06D8, 0xA2968A0E, 0x3EBA1F74, 0x4AA193A2, 0x72F8B956, 0x06E33580, 0x9ACFA0FA, 0xEED42C2C,
    0x605FDDAE, 0x14445178, 0x8868C402, 0xFC7348D4, 0xC42A6220, 0xB031EEF6, 0x2C1D7B8C, 0x5806F75A,
    0x5CAF2E64, 0x28B4A2B2, 0xB49837C8, 0xC083BB1E, 0xF8DA91EA, 0x8CC11D3C, 0x10ED8846, 0x64F60490,
    0x19BE3A3A, 0x6DA5B6EC, 0xF1892396, 0x8592AF40, 0xBDCB85B4, 0xC9D00962, 0x55FC9C18, 0x21E710CE,
    0x254EC9F0, 0x51554526, 0xCD79D05C, 0xB9625C8A, 0x813B767E, 0xF520FAA8, 0x690C6FD2, 0x1D17E304,
    0x5323A9DA, 0x2738250C, 0xBB14B076, 0xCF0F3CA0, 0xF7561654, 0x834D9A82, 0x1F610FF8, 0x6B7A832E,
    0x6FD35A10, 0x1BC8D6C6, 0x87E443BC, 0xF3FFCF6A, 0xCBA6E59E, 0xBFBD6948, 0x2391FC32, 0x578A70E4,
    0x2AC24E4E, 0x5ED9C298, 0xC2F557E2, 0xB6EEDB34, 0x8EB7F1C0, 0xFAAC7D16, 0x6680E86C, 0x129B64BA,
    0x1632BD84, 0x62293152, 0xFE05A428, 0x8A1E28FE, 0xB247020A, 0xC65C8EDC, 0x5A701BA6, 0x2E6B9770,
    0xA0E066F2, 0xD4FBEA24, 0x48D77F5E, 0x3CCCF388, 0x0495D97C, 0x708E55AA, 0xECA2C0D0, 0x98B94C06,
    0x9C109538, 0xE80B19EE, 0x74278C94, 0x003C0042, 0x38652AB6, 0x4C7EA660, 0xD052331A, 0xA449BFCC,
    0xD9018166, 0xAD1A0DB0, 0x313698CA, 0x452D141C, 0x7D743EE8, 0x096FB23E, 0x95432744, 0xE158AB92,
    0xE5F172AC, 0x91EAFE7A, 0x0DC66B00, 0x79DDE7D6, 0x4184CD22, 0x359F41F4, 0xA9B3D48E, 0xDDA85858,
    0xC0BFBB5C, 0xB4A4378A, 0x2888A2F0, 0x5C932E26, 0x64CA04D2, 0x10D18804, 0x8CFD1D7E, 0xF8E691A8,
    0xFC4F4896, 0x8854C440, 0x1478513A, 0x6063DDEC, 0x583AF718, 0x2C217BCE, 0xB00DEEB4, 0xC4166262,
    0xB95E5CC8, 0xCD45D01E, 0x51694564, 0x2572C9B2, 0x1D2BE346, 0x69306F90, 0xF51CFAEA, 0x8107763C,
    0x85AEAF02, 0xF1B523D4, 0x6D99B6AE, 0x19823A78, 0x21DB108C, 0x55C09C5A, 0xC9EC0920, 0xBDF785F6,
    0x337C7474, 0x4767F8A2, 0xDB4B6DD8, 0xAF50E10E, 0x9709CBFA, 0xE312472C, 0x7F3ED256, 0x0B255E80,
    0x0F8C87BE, 0x7B970B68, 0xE7BB9E12, 0x93A012C4, 0xABF93830, 0xDFE2B4E6, 0x43CE219C, 0x37D5AD4A,
    0x4A9D93E0, 0x3E861F36, 0xA2AA8A4C, 0xD6B1069A, 0xEEE82C6E, 0x9AF3A0B8, 0x06DF35C2, 0x72C4B914,
    0x766D602A, 0x0276ECFC, 0x9E5A7986, 0xEA41F550, 0xD218DFA4, 0xA6035372, 0x3A2FC608, 0x4E344ADE
};

unsigned int getCRC32(daal::byte *input, unsigned int prevRes, size_t len) {
    size_t i;
    daal::byte *p;

    unsigned int res, highDigit, nextDigit;
    const unsigned int crcPoly = 0xBA0DC66B;

    p = input;

    res = prevRes;

    for (i = 0; i < len; i++) {
        highDigit = res >> 24;
        nextDigit = (unsigned int)(p[len - 1 - i]);
        res = (res << 8) ^ nextDigit;
        res = res ^ crcRem[highDigit];
    }

    if (res >= crcPoly) {
        res = res ^ crcPoly;
    }

    return res;
}

void printALSRatings(daal::data_management::NumericTablePtr usersOffsetTable,
                     daal::data_management::NumericTablePtr itemsOffsetTable,
                     daal::data_management::NumericTablePtr ratings) {
    using namespace daal::data_management;

    size_t nUsers = ratings->getNumberOfRows();
    size_t nItems = ratings->getNumberOfColumns();

    BlockDescriptor<DAAL_DATA_TYPE> block1;
    ratings->getBlockOfRows(0, nUsers, readOnly, block1);
    DAAL_DATA_TYPE *ratingsData = block1.getBlockPtr();

    size_t usersOffset, itemsOffset;
    BlockDescriptor<int> block;
    usersOffsetTable->getBlockOfRows(0, 1, readOnly, block);
    usersOffset = (size_t)((block.getBlockPtr())[0]);
    usersOffsetTable->releaseBlockOfRows(block);

    itemsOffsetTable->getBlockOfRows(0, 1, readOnly, block);
    itemsOffset = (size_t)((block.getBlockPtr())[0]);
    itemsOffsetTable->releaseBlockOfRows(block);

    std::cout << " User ID, Item ID, rating" << std::endl;
    for (size_t i = 0; i < nUsers; i++) {
        for (size_t j = 0; j < nItems; j++) {
            std::cout << i + usersOffset << ", " << j + itemsOffset << ", "
                      << ratingsData[i * nItems + j] << std::endl;
        }
    }
    ratings->releaseBlockOfRows(block1);
}

#endif
