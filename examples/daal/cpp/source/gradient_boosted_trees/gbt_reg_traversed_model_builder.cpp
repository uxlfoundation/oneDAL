/* file: gbt_reg_traversed_model_builder.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
* Copyright contributors to the oneDAL project
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
!    C++ example of gradient boosted trees regression model
!    building from traversed gradient boosted trees regression model.
!
!    The program trains the gradient boosted trees regression model, gets
!    pre-computed values from nodes of each tree using traversal and build
!    model of the gradient boosted trees regression via Model Builder and
!    computes regression for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-GBT_REG_TRAVERSED_MODEL_BUILDER"></a>
 * \example gbt_reg_traversed_model_builder.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::gbt::regression;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/df_regression_train.csv";
const std::string testDatasetFileName = "data/df_regression_test.csv";
const size_t categoricalFeaturesIndices[] = { 3 };
const size_t nFeatures = 13; /* Number of features in training and testing data sets */

/* Gradient boosted trees training parameters */
const size_t maxIterations = 40;
size_t nTrees = 0;

/** Node structure for representing nodes in trees after traversing DAAL model */
struct Node {
    Node *left;
    Node *right;
    double response;
    size_t featureIndex;
    double featureValue;
    Node(double rs, size_t fi, double fv)
            : left(NULL),
              right(NULL),
              response(rs),
              featureIndex(fi),
              featureValue(fv) {}
    Node() : left(NULL), right(NULL), response(0), featureIndex(0), featureValue(0) {}
};

/** Tree structure for representing tree after traversing DAAL model */
struct Tree {
    Node *root;
    size_t nNodes;
    ~Tree() {
        if (root) {
            delete root;
        }
    }
};

/** Example of structure to remember relationship between nodes */
struct ParentPlace {
    size_t parentId;
    size_t place;
    ParentPlace(size_t _parent, size_t _place) : parentId(_parent), place(_place) {}
    ParentPlace() : parentId(0), place(0) {}
};

/** Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class BFSNodeVisitor : public daal::algorithms::tree_utils::regression::TreeNodeVisitor {
public:
    Tree *roots;
    size_t treeId;
    std::queue<Node *> parentNodes;
    virtual bool onLeafNode(
        const daal::algorithms::tree_utils::regression::LeafNodeDescriptor &desc) {
        if (desc.level == 0) {
            Node *root = roots[treeId].root;
            (*(roots + treeId)).nNodes = 1;
            root->left = NULL;
            root->right = NULL;
            root->response = desc.response;
            root->featureIndex = 0;
            root->featureValue = 0;
            treeId++;
        }
        else {
            roots[treeId - 1].nNodes++;
            Node *node = new Node(desc.response, 0, 0);

            Node *parent = parentNodes.front();
            if (parent->left == NULL) {
                parent->left = node;
            }
            else {
                parent->right = node;
                parentNodes.pop();
            }
        }
        return true;
    }

    virtual bool onSplitNode(
        const daal::algorithms::tree_utils::regression::SplitNodeDescriptor &desc) {
        if (desc.level == 0) {
            Node *root = roots[treeId].root;
            (*(roots + treeId)).nNodes = 1;
            root->left = NULL;
            root->right = NULL;
            root->response = 0;
            root->featureIndex = desc.featureIndex;
            root->featureValue = desc.featureValue;
            parentNodes.push(root);
            treeId++;
        }
        else {
            roots[treeId - 1].nNodes++;
            Node *node = new Node(0, desc.featureIndex, desc.featureValue);

            Node *parent = parentNodes.front();
            if (parent->left == NULL) {
                parent->left = node;
            }
            else {
                parent->right = node;
                parentNodes.pop();
            }
            parentNodes.push(node);
        }
        return true;
    }

    BFSNodeVisitor(size_t nTrees) : parentNodes() {
        roots = new Tree[nTrees];
        for (size_t i = 0; i < nTrees; i++) {
            roots[i].root = new Node;
        }
        treeId = 0;
    }
    ~BFSNodeVisitor() {
        if (roots)
            delete[] roots;
    }
};

training::ResultPtr trainModel();
double testModel(ModelPtr modelPtr);
void loadData(const std::string &fileName, NumericTablePtr &pData, NumericTablePtr &pDependentVar);
ModelPtr buildModel(Tree *trees);
Tree *traverseModel(ModelPtr m, BFSNodeVisitor &visitor);

bool buildTree(size_t treeId,
               Node *node,
               bool &isRoot,
               ModelBuilder &builder,
               const ParentPlace &parentPlace);

int main(int argc, char *argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    /* Train DAAL Gradient Boosted Trees Regression model */
    training::ResultPtr trainingResult = trainModel();
    std::cout << "Predict on trained model" << std::endl;
    ModelPtr trainedModel = trainingResult->get(training::model);
    if (trainedModel.get())
        nTrees = trainedModel->numberOfTrees();
    double trainedAccurcy = testModel(trainedModel);

    /* Traverse the trained model to get Tree representation */
    BFSNodeVisitor visitor(nTrees);
    Tree *trees = traverseModel(trainedModel, visitor);
    /* Build the model by ModelBuilder from Tree */
    daal::algorithms::gbt::regression::ModelPtr builtModel = buildModel(trees);
    std::cout << "Predict on built model from input user Tree " << std::endl;
    double buildModelAccurcy = testModel(builtModel);
    const char *result = (trainedAccurcy == buildModelAccurcy) ? "successfully" : "not correctly";
    std::cout << "Model was built " << result << std::endl;

    return (trainedAccurcy == buildModelAccurcy) ? 0 : 1;
}

daal::algorithms::gbt::regression::ModelPtr buildModel(Tree *trees) {
    /* create a model builder */
    ModelBuilder builder(nFeatures, nTrees);
    for (size_t i = 0; i < nTrees; i++) {
        const size_t nNodes = trees[i].nNodes;
        /* allocate the memory for certain tree */
        builder.createTree(nNodes);
        bool isRoot = true;
        /* recursive DFS traversing of certain tree with building model */
        buildTree(i, trees[i].root, isRoot, builder, ParentPlace(0, 0));
    }

    return builder.getModel();
}

bool buildTree(size_t treeId,
               Node *node,
               bool &isRoot,
               ModelBuilder &builder,
               const ParentPlace &parentPlace) {
    const int defaultLeft = 0;
    const double cover = 0.0;
    if (node->left != NULL && node->right != NULL) {
        if (isRoot) {
            ModelBuilder::NodeId parent = builder.addSplitNode(treeId,
                                                               ModelBuilder::noParent,
                                                               0,
                                                               node->featureIndex,
                                                               node->featureValue,
                                                               defaultLeft,
                                                               cover);

            isRoot = false;
            buildTree(treeId, node->left, isRoot, builder, ParentPlace(parent, 0));
            buildTree(treeId, node->right, isRoot, builder, ParentPlace(parent, 1));
        }
        else {
            ModelBuilder::NodeId parent = builder.addSplitNode(treeId,
                                                               parentPlace.parentId,
                                                               parentPlace.place,
                                                               node->featureIndex,
                                                               node->featureValue,
                                                               defaultLeft,
                                                               cover);

            buildTree(treeId, node->left, isRoot, builder, ParentPlace(parent, 0));
            buildTree(treeId, node->right, isRoot, builder, ParentPlace(parent, 1));
        }
    }
    else {
        if (isRoot) {
            builder.addLeafNode(treeId, ModelBuilder::noParent, 0, node->response, cover);
            isRoot = false;
        }
        else {
            builder.addLeafNode(treeId,
                                parentPlace.parentId,
                                parentPlace.place,
                                node->response,
                                cover);
        }
    }

    return true;
}

double testModel(daal::algorithms::gbt::regression::ModelPtr modelPtr) {
    /* Create Numeric Tables for testing data and ground truth values */
    NumericTablePtr testData;
    NumericTablePtr testGroundTruth;

    loadData(testDatasetFileName, testData, testGroundTruth);

    /* Create an algorithm object to predict values of decision forest classification */
    prediction::Batch<> algorithm;

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(prediction::data, testData);
    algorithm.input.set(prediction::model, modelPtr);

    /* Predict values of decision forest classification */
    algorithm.compute();

    /* Retrieve the algorithm results */
    NumericTablePtr prediction = algorithm.getResult()->get(prediction::prediction);
    printNumericTable(prediction, "Gradient boosted trees prediction results (first 10 rows):", 10);
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10);
    size_t nRows = 0;
    if (prediction.get())
        nRows = prediction->getNumberOfRows();
    double error = 0;
    for (size_t i = 0; i < nRows; i++) {
        error += prediction->getValue<float>(0, i) - testGroundTruth->getValue<float>(0, i);
    }

    std::cout << "Error: " << error << std::endl;
    return error;
}

training::ResultPtr trainModel() {
    /* Create Numeric Tables for training data and dependent variables */
    NumericTablePtr trainData;
    NumericTablePtr trainDependentVariable;

    loadData(trainDatasetFileName, trainData, trainDependentVariable);

    /* Create an algorithm object to train the decision forest classification model */
    training::Batch<> algorithm;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, trainData);
    algorithm.input.set(training::dependentVariable, trainDependentVariable);

    algorithm.parameter().maxIterations = maxIterations;

    /* Build the decision forest classification model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    return algorithm.getResult();
}

void loadData(const std::string &fileName, NumericTablePtr &pData, NumericTablePtr &pDependentVar) {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(fileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and dependent variables */
    pData.reset(new HomogenNumericTable<double>(nFeatures, 0, NumericTable::notAllocate));
    pDependentVar.reset(new HomogenNumericTable<double>(1, 0, NumericTable::notAllocate));
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

Tree *traverseModel(const daal::algorithms::gbt::regression::ModelPtr m, BFSNodeVisitor &visitor) {
    const size_t nTrees = m->numberOfTrees();

    for (size_t i = 0; i < nTrees; ++i) {
        m->traverseBFS(i, visitor);
    }
    return visitor.roots;
}
