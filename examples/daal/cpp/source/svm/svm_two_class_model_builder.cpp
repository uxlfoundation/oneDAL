/* file: svm_two_class_model_builder.cpp */
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
!    C++ example of two-class support vector machine (SVM) classification
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_TWO_CLASS_MODEL_BUILDER"></a>
 * \example svm_two_class_model_builder.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string trainedModelsFileName = "../data/batch/svm_two_class_trained_model.csv";

const std::string testDatasetFileName = "../data/batch/svm_two_class_test_dense.csv";

const size_t nFeatures = 20;
const float bias = -0.562F;

/* Parameters for the SVM kernel function */
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());

NumericTablePtr testGroundTruth;

void testModel(svm::ModelPtr &);
svm::ModelPtr buildModelFromTraining();

int main(int argc, char *argv[]) {
    checkArguments(argc, argv, 2, &trainedModelsFileName, &testDatasetFileName);

    svm::ModelPtr builtModel = buildModelFromTraining();
    testModel(builtModel);

    return 0;
}

svm::ModelPtr buildModelFromTraining() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve trained model .csv file */
    FileDataSource<CSVFeatureManager> modelSource(trainedModelsFileName,
                                                  DataSource::notAllocateNumericTable,
                                                  DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for supportVectors and classification coefficients */
    NumericTablePtr supportVectors =
        HomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    NumericTablePtr classificationCoefficients =
        HomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedModel =
        MergedNumericTable::create(supportVectors, classificationCoefficients);

    checkPtr(supportVectors.get());
    checkPtr(classificationCoefficients.get());

    /* Retrieve the model from input file */
    modelSource.loadDataBlock(mergedModel.get());

    size_t nSV = supportVectors->getNumberOfRows();

    svm::ModelBuilder<> modelBuilder(nFeatures, nSV);
    /* write numbers in model */
    BlockDescriptor<> blockResult;
    supportVectors->getBlockOfRows(0, nSV, readOnly, blockResult);
    float *first = blockResult.getBlockPtr();
    float *last = first + nSV * nFeatures;

    modelBuilder.setSupportVectors(first, last);

    supportVectors->releaseBlockOfRows(blockResult);

    /* set Classification Coefficients */
    classificationCoefficients->getBlockOfRows(0, nSV, readOnly, blockResult);
    first = blockResult.getBlockPtr();
    last = first + nSV;

    modelBuilder.setClassificationCoefficients(first, last);

    classificationCoefficients->releaseBlockOfRows(blockResult);

    modelBuilder.setBias(bias);

    return modelBuilder.getModel();
}

void testModel(svm::ModelPtr &inputModel) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData =
        HomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate);
    testGroundTruth = HomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate);
    NumericTablePtr mergedData = MergedNumericTable::create(testData, testGroundTruth);

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict SVM values */
    svm::prediction::Batch<float> algorithm;

    algorithm.parameter.kernel = kernel;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);

    /* Set model created externaly */
    algorithm.input.set(classifier::prediction::model, inputModel);

    /* Predict SVM values */
    algorithm.compute();

    printNumericTables<int, float>(
        testGroundTruth,
        algorithm.getResult()->get(classifier::prediction::prediction),
        "Ground truth",
        "Classification results",
        "SVM classification sample program results (first 20 observations):",
        20);
}
