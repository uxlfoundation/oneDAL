/* file: df_cls_traversed_model_builder.cpp */
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
!    C++ example of decision forest classification model traversal.
!
!    The program trains the decision forest classification model on a training
!    datasetFileName and prints the trained model by its depth-first traversing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_CLS_TRAVERSED_MODEL_BUILDER"></a>
 * \example df_cls_traversed_model_builder.cpp
 */

#include "daal.h"
#include "service.h"
#include <stack>
#include <map>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::decision_forest::classification;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/df_classification_train_data.csv";
const std::string trainDatasetLabelFileName = "data/df_classification_train_label.csv";
const std::string testDatasetFileName = "data/df_classification_test_data.csv";
const std::string testDatasetLabelFileName = "data/df_classification_test_label.csv";

const size_t nFeatures = 3; /* Number of features in training and testing data sets */

/* Decision forest parameters */
size_t nTrees = 10;
const size_t minObservationsInLeafNode = 8;

const size_t nClasses = 5; /* Number of classes */

/** Node structure for representing nodes in trees after traversing DAAL model */
struct Node {
    Node *left;
    Node *right;
    size_t classLabel;
    size_t featureIndex;
    double featureValue;
    Node(size_t cl, size_t fi, double fv)
            : left(NULL),
              right(NULL),
              classLabel(cl),
              featureIndex(fi),
              featureValue(fv) {}
    Node() : left(NULL), right(NULL), classLabel(0), featureIndex(0), featureValue(0) {}
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
/** Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class BFSNodeVisitor : public daal::algorithms::tree_utils::classification::TreeNodeVisitor {
public:
    Tree *roots;
    size_t treeId;
    std::queue<Node *> parentNodes;
    virtual bool onLeafNode(const tree_utils::classification::LeafNodeDescriptor &desc) {
        if (desc.level == 0) {
            Node *root = roots[treeId].root;
            (*(roots + treeId)).nNodes = 1;
            root->left = NULL;
            root->right = NULL;
            root->classLabel = desc.label;
            root->featureIndex = 0;
            root->featureValue = 0;
            treeId++;
        }
        else {
            roots[treeId - 1].nNodes++;
            Node *node = new Node(desc.label, 0, 0);

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

    virtual bool onSplitNode(const tree_utils::classification::SplitNodeDescriptor &desc) {
        if (desc.level == 0) {
            Node *root = roots[treeId].root;
            (*(roots + treeId)).nNodes = 1;
            root->left = NULL;
            root->right = NULL;
            root->classLabel = 0;
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
};

training::ResultPtr trainModel();
double testModel(daal::algorithms::decision_forest::classification::ModelPtr modelPtr);
void loadData(const std::string &fileName, NumericTablePtr &pData, NumericTablePtr &pDependentVar);
daal::algorithms::decision_forest::classification::ModelPtr buildModel(Tree *trees);
Tree *traverseModel(daal::algorithms::decision_forest::classification::ModelPtr m,
                    BFSNodeVisitor &visitor);
bool buildTree(size_t treeId,
               Node *node,
               bool &isRoot,
               ModelBuilder &builder,
               std::map<Node *, ParentPlace> &parentMap);

int main(int argc, char *argv[]) {
    checkArguments(argc,
                   argv,
                   4,
                   &trainDatasetFileName,
                   &trainDatasetLabelFileName,
                   &testDatasetFileName,
                   &testDatasetLabelFileName);

    /* train DAAL DF Classification model */
    training::ResultPtr trainingResult = trainModel();

    std::cout << "Predict on trained model" << std::endl;
    auto trainedModel = trainingResult->get(classifier::training::model);
    if (trainedModel.get())
        nTrees = trainedModel->numberOfTrees();

    double trainedAccurcy = testModel(trainedModel);

    /* traverse the trained model to get Tree representation */
    BFSNodeVisitor visitor(nTrees);
    Tree *trees = traverseModel(trainedModel, visitor);
    /* build the model by ModelBuilder from Tree */
    daal::algorithms::decision_forest::classification::ModelPtr builtModel = buildModel(trees);
    std::cout << "Predict on built model from input user Tree " << std::endl;
    double buildModelAccurcy = testModel(builtModel);
    if (trainedAccurcy == buildModelAccurcy) {
        std::cout << "Model was built successfully" << std::endl;
    }
    else {
        std::cout << "Model was built not correctly" << std::endl;
        return 1;
    }
    return 0;
}

daal::algorithms::decision_forest::classification::ModelPtr buildModel(Tree *trees) {
    /* create a model builder */
    ModelBuilder builder(nClasses, nTrees);
    /* map to get relationship between nodes */
    std::map<Node *, ParentPlace> parentMap;
    for (size_t i = 0; i < nTrees; i++) {
        const size_t nNodes = trees[i].nNodes;
        /* allocate the memory for certain tree */
        builder.createTree(nNodes);
        bool isRoot = true;
        /* recursive DFS traversing of certain tree with building model */
        buildTree(i, trees[i].root, isRoot, builder, parentMap);
        parentMap.erase(parentMap.begin(), parentMap.end());
    }
    builder.setNFeatures(nFeatures);
    return builder.getModel();
}

bool buildTree(size_t treeId,
               Node *node,
               bool &isRoot,
               ModelBuilder &builder,
               std::map<Node *, ParentPlace> &parentMap) {
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

            parentMap[node->left] = ParentPlace(parent, 0);
            ;
            parentMap[node->right] = ParentPlace(parent, 1);
            ;
            isRoot = false;
        }
        else {
            ParentPlace p = parentMap[node];
            ModelBuilder::NodeId parent = builder.addSplitNode(treeId,
                                                               p.parentId,
                                                               p.place,
                                                               node->featureIndex,
                                                               node->featureValue,
                                                               defaultLeft,
                                                               cover);

            parentMap[node->left] = ParentPlace(parent, 0);
            ;
            parentMap[node->right] = ParentPlace(parent, 1);
            ;
        }
    }
    else {
        if (isRoot) {
            builder.addLeafNode(treeId, ModelBuilder::noParent, 0, node->classLabel, cover);
            isRoot = false;
        }
        else {
            ParentPlace p = parentMap[node];
            builder.addLeafNode(treeId, p.parentId, p.place, node->classLabel, cover);
        }
        return true;
    }
    buildTree(treeId, node->left, isRoot, builder, parentMap);
    buildTree(treeId, node->right, isRoot, builder, parentMap);
    return true;
}

double testModel(daal::algorithms::decision_forest::classification::ModelPtr modelPtr) {
    /* Create Numeric Tables for testing data and ground truth values */
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from
     * a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::doAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    testDataSource.loadDataBlock();
    FileDataSource<CSVFeatureManager> testLabelSource(testDatasetLabelFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    testLabelSource.loadDataBlock();

    /* Create an algorithm object to predict values of decision forest classification */
    prediction::Batch<> algorithm(nClasses);

    /* Pass a testing data set and the trained model to the algorithm */
    algorithm.input.set(daal::algorithms::classifier::prediction::data,
                        testDataSource.getNumericTable());
    algorithm.input.set(daal::algorithms::classifier::prediction::model, modelPtr);

    algorithm.parameter().votingMethod = prediction::unweighted;
    /* Predict values of decision forest classification */
    algorithm.compute();

    /* Retrieve the algorithm results */
    NumericTablePtr prediction = algorithm.getResult()->get(classifier::prediction::prediction);
    printNumericTable(prediction, "Decision forest prediction results (first 10 rows):", 10);
    printNumericTable(testLabelSource.getNumericTable(), "Ground truth (first 10 rows):", 10);
    const size_t nRows = prediction->getNumberOfRows();
    size_t countOfNotCorrect = 0;
    for (size_t i = 0; i < nRows; i++) {
        if (prediction->getValue<float>(0, i) !=
            testLabelSource.getNumericTable()->getValue<float>(0, i))
            countOfNotCorrect++;
    }
    double accuracy = 1 - double(countOfNotCorrect) / nRows;
    std::cout << "Accuracy: " << accuracy << std::endl;
    return accuracy;
}

training::ResultPtr trainModel() {
    /* Create Numeric Tables for training data and dependent variables */
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data
     * from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    FileDataSource<CSVFeatureManager> trainLabelSource(trainDatasetLabelFileName,
                                                       DataSource::doAllocateNumericTable,
                                                       DataSource::doDictionaryFromContext);
    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock();
    trainLabelSource.loadDataBlock();

    /* Create an algorithm object to train the decision forest classification model */
    training::Batch<> algorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(daal::algorithms::classifier::training::data,
                        trainDataSource.getNumericTable());
    algorithm.input.set(daal::algorithms::classifier::training::labels,
                        trainLabelSource.getNumericTable());

    algorithm.parameter().nTrees = nTrees;
    algorithm.parameter().featuresPerNode = nFeatures;
    algorithm.parameter().minObservationsInLeafNode = minObservationsInLeafNode;

    /* Build the decision forest classification model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    return algorithm.getResult();
}

Tree *traverseModel(const daal::algorithms::decision_forest::classification::ModelPtr m,
                    BFSNodeVisitor &visitor) {
    const size_t nTrees = m->getNumberOfTrees();
    for (size_t i = 0; i < nTrees; ++i) {
        m->traverseBFS(i, visitor);
    }
    return visitor.roots;
}
