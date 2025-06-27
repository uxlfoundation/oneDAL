/* file: df_cls_dense_batch_model_builder.cpp */
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
!    C++ example of decision forest classification model building.
!
!    The program builds the decision forest classification model
!     via Model Builder and computes classification for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_CLS_MODEL_BUILDER"></a>
 * \example df_cls_dense_batch_model_builder.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::decision_forest::classification;

/* Input data set parameters */
const std::string testDatasetFileName = "../data/batch/df_classification_model_builder_test.csv";
const size_t categoricalFeaturesIndices[] = { 2 };
const size_t nFeatures = 3; /* Number of features in training and testing data sets */

/* Decision forest parameters */
const size_t nTrees = 3;
const size_t nClasses = 5; /* Number of classes */

void testModel(decision_forest::classification::ModelPtr& model);
decision_forest::classification::ModelPtr buildModel();
void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &testDatasetFileName);

    decision_forest::classification::ModelPtr model = buildModel();
    testModel(model);

    return 0;
}

decision_forest::classification::ModelPtr buildModel() {
    const size_t nNodes = 3;
    const int defaultLeft = 0;
    const double cover = 0.0;

    ModelBuilder modelBuilder(nClasses, nTrees);
    ModelBuilder::TreeId tree1 = modelBuilder.createTree(nNodes);
    ModelBuilder::NodeId root1 =
        modelBuilder
            .addSplitNode(tree1, ModelBuilder::noParent, 0, 0, 0.174108, defaultLeft, cover);
    /* ModelBuilder::NodeId child12 = */ modelBuilder.addLeafNode(tree1, root1, 1, 4, cover);
    double proba11[] = { 0.8, 0.1, 0.0, 0.1, 0.0 };
    /* ModelBuilder::NodeId child11 = */ modelBuilder.addLeafNodeByProba(tree1,
                                                                         root1,
                                                                         0,
                                                                         proba11,
                                                                         cover);

    ModelBuilder::TreeId tree2 = modelBuilder.createTree(nNodes);
    ModelBuilder::NodeId root2 =
        modelBuilder
            .addSplitNode(tree2, ModelBuilder::noParent, 0, 1, 0.571184, defaultLeft, cover);
    /* ModelBuilder::NodeId child22 = */ modelBuilder.addLeafNode(tree2, root2, 1, 4, cover);
    /* ModelBuilder::NodeId child21 = */ modelBuilder.addLeafNode(tree2, root2, 0, 2, cover);

    ModelBuilder::TreeId tree3 = modelBuilder.createTree(nNodes);
    ModelBuilder::NodeId root3 =
        modelBuilder
            .addSplitNode(tree3, ModelBuilder::noParent, 0, 0, 0.303995, defaultLeft, cover);
    double proba32[] = { 0.05, 0.1, 0.0, 0.1, 0.75 };
    /* ModelBuilder::NodeId child32 = */ modelBuilder.addLeafNodeByProba(tree3,
                                                                         root3,
                                                                         1,
                                                                         proba32,
                                                                         cover);
    /* ModelBuilder::NodeId child31 = */ modelBuilder.addLeafNode(tree3, root3, 0, 2, cover);
    modelBuilder.setNFeatures(nFeatures);
    return modelBuilder.getModel();
}

void testModel(decision_forest::classification::ModelPtr& model) {
    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData;
    NumericTablePtr testGroundTruth;

    loadData(testDatasetFileName, testData, testGroundTruth);

    /* Create an algorithm object to predict values of decision forest classification */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);

    /* set model obtained by builder */
    algorithm.input.set(classifier::prediction::model, model);

    /* set voting method */
    algorithm.parameter().votingMethod = prediction::unweighted;

    /* Predict values of decision forest classification */
    algorithm.compute();

    /* Retrieve the algorithm results */
    classifier::prediction::ResultPtr predictionResult = algorithm.getResult();
    printNumericTable(predictionResult->get(classifier::prediction::prediction),
                      "Decision forest prediction results (first 10 rows):",
                      10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
}

void loadData(const std::string& fileName, NumericTablePtr& pData, NumericTablePtr& pDependentVar) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData.reset(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    pDependentVar.reset(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(pData, pDependentVar));

    /* Retrieve the data from input file */
    trainDataSource.loadDataBlock(mergedData.get());

    NumericTableDictionaryPtr pDictionary = pData->getDictionarySharedPtr();
    for (size_t i = 0,
                n = sizeof(categoricalFeaturesIndices) / sizeof(categoricalFeaturesIndices[0]);
         i < n;
         ++i)
        (*pDictionary)[categoricalFeaturesIndices[i]].featureType =
            data_feature_utils::DAAL_CATEGORICAL;
}
