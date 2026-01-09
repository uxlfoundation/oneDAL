/* file: df_reg_traverse_model.cpp */
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
!    C++ example of decision forest regression model traversal.
!
!    The program trains the decision forest regression model on a training
!    datasetFileName and prints the trained model by its depth-first traversing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DF_REG_TRAVERSE_MODEL"></a>
 * \example df_reg_traverse_model.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;
using namespace daal::algorithms::decision_forest::regression;

/* Input data set parameters */
const std::string trainDatasetFileName = "data/df_regression_train_data.csv";
const std::string trainDatasetLabelFileName = "data/df_regression_train_label.csv";

/* Decision forest parameters */
const size_t nTrees = 2;

training::ResultPtr trainModel();
void printModel(const daal::algorithms::decision_forest::regression::Model& m);

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &trainDatasetFileName);

    training::ResultPtr trainingResult = trainModel();
    printModel(*trainingResult->get(training::model));

    return 0;
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

    /* Create an algorithm object to train the decision forest regression model with the default method */
    training::Batch<> algorithm;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, trainDataSource.getNumericTable());
    algorithm.input.set(training::dependentVariable, trainLabelSource.getNumericTable());

    algorithm.parameter().nTrees = nTrees;

    /* Build the decision forest regression model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    return algorithm.getResult();
}

/** Visitor class implementing TreeNodeVisitor interface, prints out tree nodes of the model when it is called back by model traversal method */
class PrintNodeVisitor : public daal::algorithms::tree_utils::regression::TreeNodeVisitor {
public:
    virtual bool onLeafNode(
        const daal::algorithms::tree_utils::regression::LeafNodeDescriptor& desc) {
        for (size_t i = 0; i < desc.level; ++i)
            std::cout << "  ";
        std::cout << "Level " << desc.level << ", leaf node. Response value = " << desc.response
                  << ", Impurity = " << desc.impurity
                  << ", Number of samples = " << desc.nNodeSampleCount << std::endl;
        return true;
    }

    virtual bool onSplitNode(
        const daal::algorithms::tree_utils::regression::SplitNodeDescriptor& desc) {
        for (size_t i = 0; i < desc.level; ++i)
            std::cout << "  ";
        std::cout << "Level " << desc.level << ", split node. Feature index = " << desc.featureIndex
                  << ", feature value = " << desc.featureValue << ", Impurity = " << desc.impurity
                  << ", Number of samples = " << desc.nNodeSampleCount << std::endl;
        return true;
    }
};

void printModel(const daal::algorithms::decision_forest::regression::Model& m) {
    PrintNodeVisitor visitor;
    std::cout << "Number of trees: " << m.getNumberOfTrees() << std::endl;
    for (size_t i = 0, n = m.getNumberOfTrees(); i < n; ++i) {
        std::cout << "Tree #" << i << std::endl;
        m.traverseDFS(i, visitor);
    }
}
