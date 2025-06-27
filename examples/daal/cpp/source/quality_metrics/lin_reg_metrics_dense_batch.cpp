/* file: lin_reg_metrics_dense_batch.cpp */
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
!    C++ example of linear regression quality metrics in batch processing mode.
!
!    The program trains the linear regression model on a training
!    datasetFileName with normal equation and QR methods and computes quality
!    metrics for the models.
!******************************************************************************/

/**
* <a name="DAAL-EXAMPLE-CPP-LIN_REG_QUALITY_METRIC_SET_BATCH"></a>
* \example lin_reg_metrics_dense_batch.cpp
*/

#include "daal.h"
#include "service.h"
#include "algorithms/linear_regression/linear_regression_quality_metric_set_batch.h"
#include "algorithms/linear_regression/linear_regression_quality_metric_set_types.h"
#include "algorithms/linear_regression/linear_regression_single_beta_batch.h"
#include "algorithms/linear_regression/linear_regression_single_beta_types.h"
#include "algorithms/linear_regression/linear_regression_group_of_betas_types.h"
#include "algorithms/linear_regression/linear_regression_group_of_betas_batch.h"

using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::linear_regression;
using namespace daal::algorithms::linear_regression::quality_metric;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/linear_regression_train.csv";
const std::string testDatasetFileName = "../data/batch/linear_regression_test.csv";

const size_t nFeatures = 10;
const size_t nDependentVariables = 2;

template <class TrainingAlgorithm>
void trainModel(TrainingAlgorithm& algo);
void testModelQuality();
void printResults();

training::ResultPtr trainingResult;
quality_metric_set::ResultCollectionPtr qmsResult;
NumericTablePtr trainData;
NumericTablePtr trainDependentVariables;

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    trainData =
        NumericTablePtr(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    trainDependentVariables = NumericTablePtr(
        new HomogenNumericTable<>(nDependentVariables, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainDependentVariables));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());

    for (size_t i = 0; i < 2; ++i) {
        if (i == 0) {
            /* Create an algorithm object to train the multiple linear regression model with normal equation method */
            std::cout << "Train model with normal equation algorithm." << std::endl;
            training::Batch<> algorithm;
            trainModel(algorithm);
        }
        else {
            /* Create an algorithm object to train the multiple linear regression model with QR method */
            std::cout << "Train model with QR algorithm." << std::endl;
            training::Batch<float, training::qrDense> algorithm;
            trainModel(algorithm);
        }
        testModelQuality();
        printResults();
    }
    return 0;
}

template <class TrainingAlgorithm>
void trainModel(TrainingAlgorithm& algorithm) {
    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, trainData);
    algorithm.input.set(training::dependentVariables, trainDependentVariables);

    /* Build the multiple linear regression model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
    printNumericTable(trainingResult->get(training::model)->getBeta(),
                      "Linear Regression coefficients:");
}

NumericTablePtr predictResults(NumericTablePtr& data) {
    /* Create an algorithm object to predict values of multiple linear regression */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, data);
    algorithm.input.set(prediction::model, trainingResult->get(training::model));

    /* Predict values of multiple linear regression */
    algorithm.compute();

    /* Retrieve the algorithm results */
    prediction::ResultPtr predictionResult = algorithm.getResult();
    return predictionResult->get(prediction::prediction);
}

NumericTablePtr predictReducedModelResults() {
    ModelPtr model = trainingResult->get(training::model);

    NumericTablePtr betas = model->getBeta();
    const size_t nBetas = model->getNumberOfBetas();

    /* Set beta coefficients #2 and #10 to zero */
    size_t j1 = 2;
    size_t j2 = 10;
    float* savedBeta = new float[nBetas * nDependentVariables];
    {
        BlockDescriptor<> block;
        betas->getBlockOfRows(0, nDependentVariables, readWrite, block);
        float* pBeta = block.getBlockPtr();
        for (size_t i = 0; i < nDependentVariables; ++i) {
            savedBeta[nBetas * i + j1] = pBeta[nBetas * i + j1];
            savedBeta[nBetas * i + j2] = pBeta[nBetas * i + j2];
            pBeta[nBetas * i + j1] = 0;
            pBeta[nBetas * i + j2] = 0;
        }
        betas->releaseBlockOfRows(block);
    }

    /* Predict with reduced model */
    NumericTablePtr predictedResults = predictResults(trainData);

    /* Restore the coefficients */
    {
        BlockDescriptor<> block;
        betas->getBlockOfRows(0, nDependentVariables, readWrite, block);
        float* pBeta = block.getBlockPtr();
        for (size_t i = 0; i < nDependentVariables; ++i) {
            pBeta[nBetas * i + j1] = savedBeta[nBetas * i + j1];
            pBeta[nBetas * i + j2] = savedBeta[nBetas * i + j2];
        }
        betas->releaseBlockOfRows(block);
    }
    delete[] savedBeta;
    return predictedResults;
}

void testModelQuality() {
    /* Predict results with the full model */
    NumericTablePtr predictedResults = predictResults(trainData);
    printNumericTable(trainDependentVariables, "Expected responses (first 20 rows):", 20);
    printNumericTable(predictedResults, "Predicted responses (first 20 rows):", 20);

    ModelPtr model = trainingResult->get(training::model);

    /* Predict results with the reduced model */
    NumericTablePtr predictedReducedModelResults = predictReducedModelResults();
    printNumericTable(predictedReducedModelResults,
                      "Responses predicted with reduced model (first 20 rows):",
                      20);

    /* Create a quality metric set object to compute quality metrics of the linear regression algorithm */
    const size_t nBetaReducedModel = model->getNumberOfBetas() - 2;
    quality_metric_set::Batch qualityMetricSet(model->getNumberOfBetas(), nBetaReducedModel);

    /* Set input for a single beta metrics algorithm */
    algorithms::InputPtr algInput =
        qualityMetricSet.getInputDataCollection()->getInput(quality_metric_set::singleBeta);
    single_beta::InputPtr singleBeta = single_beta::Input::cast(algInput);

    singleBeta->set(single_beta::expectedResponses, trainDependentVariables);
    singleBeta->set(single_beta::predictedResponses, predictedResults);
    singleBeta->set(single_beta::model, model);

    /* Set input for a group of betas metrics algorithm */
    algInput =
        qualityMetricSet.getInputDataCollection()->getInput(quality_metric_set::groupOfBetas);
    group_of_betas::InputPtr groupOfBetas = group_of_betas::Input::cast(algInput);
    groupOfBetas->set(group_of_betas::expectedResponses, trainDependentVariables);
    groupOfBetas->set(group_of_betas::predictedResponses, predictedResults);
    groupOfBetas->set(group_of_betas::predictedReducedModelResponses, predictedReducedModelResults);

    /* Compute quality metrics */
    qualityMetricSet.compute();

    /* Retrieve the quality metrics */
    qmsResult = qualityMetricSet.getResultCollection();
}

void printResults() {
    /* Print the quality metrics for a single beta */
    std::cout << "Quality metrics for a single beta" << std::endl;
    {
        single_beta::ResultPtr result =
            single_beta::Result::cast(qmsResult->getResult(quality_metric_set::singleBeta));
        if (!result)
            return;
        printNumericTable(result->get(single_beta::rms),
                          "Root means square errors for each response (dependent variable):");
        printNumericTable(result->get(single_beta::variance),
                          "Variance for each response (dependent variable):");
        printNumericTable(result->get(single_beta::zScore), "Z-score statistics:");
        printNumericTable(result->get(single_beta::confidenceIntervals),
                          "Confidence intervals for each beta coefficient:");
        printNumericTable(result->get(single_beta::inverseOfXtX), "Inverse(Xt * X) matrix:");

        data_management::DataCollectionPtr coll = result->get(single_beta::betaCovariances);
        for (size_t i = 0; i < coll->size(); ++i) {
            std::ostringstream str;
            str << "Variance-covariance matrix for betas of " << i << "-th response" << std::endl;
            NumericTablePtr betaCov = data_management::NumericTable::cast((*coll)[i]);
            printNumericTable(betaCov, str.str().c_str());
        }
    }
    /* Print quality metrics for a group of betas */
    std::cout << "Quality metrics for a group of betas" << std::endl;
    {
        group_of_betas::ResultPtr result =
            group_of_betas::Result::cast(qmsResult->getResult(quality_metric_set::groupOfBetas));
        if (!result)
            return;

        printNumericTable(result->get(group_of_betas::expectedMeans),
                          "Means of expected responses for each dependent variable:",
                          0,
                          0,
                          20);
        printNumericTable(result->get(group_of_betas::expectedVariance),
                          "Variance of expected responses for each dependent variable:",
                          0,
                          0,
                          20);
        printNumericTable(result->get(group_of_betas::regSS),
                          "Regression sum of squares of expected responses:",
                          0,
                          0,
                          20);
        printNumericTable(result->get(group_of_betas::resSS),
                          "Sum of squares of residuals for each dependent variable:",
                          0,
                          0,
                          20);
        printNumericTable(result->get(group_of_betas::tSS),
                          "Total sum of squares for each dependent variable:",
                          0,
                          0,
                          20);
        printNumericTable(result->get(group_of_betas::determinationCoeff),
                          "Determination coefficient for each dependent variable:",
                          0,
                          0,
                          20);
        printNumericTable(result->get(group_of_betas::fStatistics),
                          "F-statistics for each dependent variable:",
                          0,
                          0,
                          20);
    }
}
