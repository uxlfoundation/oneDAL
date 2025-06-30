/* file: svm_multi_class_model_builder.cpp */
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
!    C++ example of multi-class support vector machine (SVM) classification
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SVM_MULTI_CLASS_MODEL_BUILDER"></a>
 * \example svm_multi_class_model_builder.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string trainedModelsFileNames[] = {
    "../data/batch/svm_multi_class_trained_model_01.csv",
    "../data/batch/svm_multi_class_trained_model_02.csv",
    "../data/batch/svm_multi_class_trained_model_12.csv"
};
float biases[] = { -0.774F, -1.507F, -7.559F };

const std::string testDatasetFileName = "../data/batch/multiclass_iris_train.csv";

const size_t nFeatures = 4;
const size_t nClasses = 3;

services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());

classifier::prediction::ResultPtr predictionResult;
kernel_function::KernelIfacePtr kernel(new kernel_function::linear::Batch<>());
NumericTablePtr testGroundTruth;

multi_class_classifier::ModelPtr buildModelFromTraining();
void testModel(multi_class_classifier::ModelPtr& inputModel);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &testDatasetFileName);

    multi_class_classifier::ModelPtr builtModel = buildModelFromTraining();
    prediction->parameter.kernel = kernel;
    testModel(builtModel);
    return 0;
}

multi_class_classifier::ModelPtr buildModelFromTraining() {
    multi_class_classifier::ModelBuilder<> multiBuilder(nFeatures, nClasses);

    size_t imodel = 0;
    for (size_t iClass = 1; iClass < nClasses; iClass++) {
        for (size_t jClass = 0; jClass < iClass; jClass++, imodel++) {
            /* Initialize FileDataSource<CSVFeatureManager> to retrieve the binary classifications models */
            FileDataSource<CSVFeatureManager> modelSource(trainedModelsFileNames[imodel],
                                                          DataSource::doAllocateNumericTable,
                                                          DataSource::doDictionaryFromContext);

            /* Create Numeric Tables for support vectors and classification coeffes */
            NumericTablePtr supportVectors(
                new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
            NumericTablePtr classificationCoefficients(
                new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
            NumericTablePtr mergedModel(
                new MergedNumericTable(supportVectors, classificationCoefficients));

            /* Retrieve the data from input file */
            modelSource.loadDataBlock(mergedModel.get());

            float bias = biases[imodel];
            size_t nSV = supportVectors->getNumberOfRows();

            /* write numbers in model */
            BlockDescriptor<> blockResult;
            supportVectors->getBlockOfRows(0, nSV, readOnly, blockResult);
            float* first = blockResult.getBlockPtr();
            float* last = first + nSV * nFeatures;

            svm::ModelBuilder<> modelBuilder(nFeatures, nSV);
            /* set support vectors */
            modelBuilder.setSupportVectors(first, last);
            supportVectors->releaseBlockOfRows(blockResult);

            /* set Classification Coefficients */
            classificationCoefficients->getBlockOfRows(0, nSV, readOnly, blockResult);
            first = blockResult.getBlockPtr();
            last = first + nSV;

            modelBuilder.setClassificationCoefficients(first, last);

            classificationCoefficients->releaseBlockOfRows(blockResult);

            modelBuilder.setBias(bias);

            multiBuilder.setTwoClassClassifierModel(jClass, iClass, modelBuilder.getModel());
        }
    }

    return multiBuilder.getModel();
}

void testModel(multi_class_classifier::ModelPtr& inputModel) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create an algorithm object to predict multi-class SVM values */
    multi_class_classifier::prediction::Batch<> algorithm(nClasses);

    //algorithm.parameter.training = training;
    algorithm.parameter.prediction = prediction;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model, inputModel);

    /* Predict multi-class SVM values */
    algorithm.compute();

    /* Retrieve the algorithm results */
    predictionResult = algorithm.getResult();

    printNumericTables<int, int>(
        testGroundTruth,
        predictionResult->get(classifier::prediction::prediction),
        "Ground truth",
        "Classification results",
        "Multi-class SVM classification sample program results (first 20 observations):",
        20);
}
