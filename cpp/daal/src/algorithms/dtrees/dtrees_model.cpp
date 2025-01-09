/* file: dtrees_model.cpp */
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
//++
//  Implementation of the class defining the decision trees model
//--
*/

#include "src/algorithms/dtrees/dtrees_model_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace internal
{
Tree::~Tree() {}

ModelImpl::ModelImpl() : _nTree(0) {}

ModelImpl::ModelImpl(const ModelImpl & other)
{
    const size_t nTree = other._nTree.get();
    resize(nTree); // sets _nTree = 0
    _nTree.set(nTree);
    for (size_t i = 0; i < nTree; ++i)
    {
        (*_serializationData)[i] = (*other._serializationData)[i];
        (*_impurityTables)[i]    = (*other._impurityTables)[i];
        (*_nNodeSampleTables)[i] = (*other._nNodeSampleTables)[i];
        (*_probTbl)[i]           = (*other._probTbl)[i];
    }
}

ModelImpl & ModelImpl::operator=(const ModelImpl & other)
{
    if (this != &other)
    {
        destroy();
        const size_t nTree = other._nTree.get();
        resize(nTree); // sets _nTree = 0
        _nTree.set(nTree);
        for (size_t i = 0; i < nTree; ++i)
        {
            (*_serializationData)[i] = (*other._serializationData)[i];
            (*_impurityTables)[i]    = (*other._impurityTables)[i];
            (*_nNodeSampleTables)[i] = (*other._nNodeSampleTables)[i];
            (*_probTbl)[i]           = (*other._probTbl)[i];
        }
    }
    return *this;
}

ModelImpl::~ModelImpl()
{
    destroy();
}

void ModelImpl::destroy()
{
    _serializationData.reset();
    _impurityTables.reset();
    _nNodeSampleTables.reset();
    _probTbl.reset();
}

bool ModelImpl::reserve(const size_t nTrees)
{
    if (_serializationData.get()) return false;
    _nTree.set(0);
    _serializationData.reset(new DataCollection());
    _serializationData->resize(nTrees);

    _impurityTables.reset(new DataCollection());
    _impurityTables->resize(nTrees);

    _nNodeSampleTables.reset(new DataCollection());
    _nNodeSampleTables->resize(nTrees);

    _probTbl.reset(new DataCollection());
    _probTbl->resize(nTrees);

    return _serializationData.get();
}

bool ModelImpl::resize(const size_t nTrees)
{
    if (_serializationData.get()) return false;
    _nTree.set(0);
    _serializationData.reset(new DataCollection(nTrees));
    _impurityTables.reset(new DataCollection(nTrees));
    _nNodeSampleTables.reset(new DataCollection(nTrees));
    _probTbl.reset(new DataCollection(nTrees));
    return _serializationData.get();
}

void ModelImpl::clear()
{
    if (_serializationData.get()) _serializationData.reset();

    if (_impurityTables.get()) _impurityTables.reset();

    if (_nNodeSampleTables.get()) _nNodeSampleTables.reset();

    if (_probTbl.get()) _probTbl.reset();

    _nTree.set(0);
}

void MemoryManager::destroy()
{
    for (size_t i = 0; i < _aChunk.size(); ++i)
    {
        daal_free(_aChunk[i]);
        _aChunk[i] = nullptr;
    }
    _aChunk.clear();
    _posInChunk = 0;
    _iCurChunk  = -1;
}

void * MemoryManager::alloc(size_t nBytes)
{
    DAAL_ASSERT(nBytes <= _chunkSize);
    size_t pos = 0; //pos in the chunk to allocate from
    if ((_iCurChunk >= 0) && (_posInChunk + nBytes <= _chunkSize))
    {
        //allocate from the current chunk
        pos = _posInChunk;
    }
    else
    {
        if (!_aChunk.size() || _iCurChunk + 1 >= _aChunk.size())
        {
            //allocate a new chunk
            DAAL_ASSERT(_aChunk.size() ? _iCurChunk >= 0 : _iCurChunk < 0);
            byte * ptr = (byte *)services::daal_calloc(_chunkSize);
            if (!ptr) return nullptr;
            _aChunk.push_back(ptr);
        }
        //there are free chunks, make next available a current one and allocate from it
        _iCurChunk++;
        pos         = 0;
        _posInChunk = 0;
    }
    //allocate from the current chunk
    _posInChunk += nBytes;
    return _aChunk[_iCurChunk] + pos;
}

void MemoryManager::reset()
{
    _iCurChunk  = -1;
    _posInChunk = 0;
}

services::Status createTreeInternal(data_management::DataCollectionPtr & serializationData, size_t nNodes, size_t & resId)
{
    if (nNodes == 0)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    services::Status s;

    size_t treeId            = 0;
    bool isNotEmptyTreeTable = ((*(serializationData))[treeId].get()) != nullptr;
    const size_t nTrees      = (*(serializationData)).size();
    while (isNotEmptyTreeTable && (treeId < nTrees))
    {
        isNotEmptyTreeTable = ((*(serializationData))[treeId].get()) != nullptr;
        if (isNotEmptyTreeTable) treeId++;
    }
    if (treeId == nTrees)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }

    services::SharedPtr<DecisionTreeTable> treeTablePtr(new DecisionTreeTable(nNodes));
    const size_t nRows              = treeTablePtr->getNumberOfRows();
    DecisionTreeNode * const pNodes = (DecisionTreeNode *)treeTablePtr->getArray();
    DAAL_CHECK_MALLOC(pNodes)
    pNodes[0].featureIndex           = __NODE_RESERVED_ID;
    pNodes[0].leftIndexOrClass       = 0;
    pNodes[0].featureValueOrResponse = 0;
    for (size_t i = 1; i < nRows; i++)
    {
        pNodes[i].featureIndex           = __NODE_FREE_ID;
        pNodes[i].leftIndexOrClass       = 0;
        pNodes[i].featureValueOrResponse = 0;
    }
    (*(serializationData))[treeId] = treeTablePtr;
    resId                          = treeId;
    return s;
}

void setNode(DecisionTreeNode & node, int featureIndex, size_t classLabel, double cover)
{
    node.featureIndex           = featureIndex;
    node.leftIndexOrClass       = classLabel;
    node.cover                  = cover;
    node.featureValueOrResponse = 0;
}

void setNode(DecisionTreeNode & node, int featureIndex, double response, double cover)
{
    node.featureIndex           = featureIndex;
    node.leftIndexOrClass       = 0;
    node.cover                  = cover;
    node.featureValueOrResponse = response;
}

void setProbabilities(const size_t treeId, const size_t nodeId, const size_t response, const data_management::DataCollectionPtr probTbl,
                      const double * const prob)
{
    if (probTbl.get() == nullptr)
    {
        return;
    }
    const auto treeProbaTable = (const data_management::HomogenNumericTable<double> *)(*probTbl)[treeId].get();
    const size_t nClasses     = treeProbaTable->getNumberOfRows();
    double * const probOfTree = treeProbaTable->getArray() + nodeId * nClasses;
    if (prob != nullptr)
    {
        for (size_t classIndex = 0; classIndex < nClasses; ++classIndex)
        {
            probOfTree[classIndex] = prob[classIndex];
        }
    }
    else
    {
        for (size_t classIndex = 0; classIndex < nClasses; ++classIndex)
        {
            probOfTree[classIndex] = 0.0;
        }
        probOfTree[response] = 1.0;
    }
}

services::Status addSplitNodeInternal(data_management::DataCollectionPtr & serializationData, size_t treeId, size_t parentId, size_t position,
                                      size_t featureIndex, double featureValue, int defaultLeft, double cover, size_t & res)
{
    const size_t noParent = static_cast<size_t>(-1);
    services::Status s;

    if ((treeId > (*(serializationData)).size()) || (position != 0 && position != 1))
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }

    const DecisionTreeTable * const pTreeTable = static_cast<DecisionTreeTable *>((*(serializationData))[treeId].get());
    if (!pTreeTable) return services::Status(services::ErrorID::ErrorNullPtr);
    const size_t nRows             = pTreeTable->getNumberOfRows();
    DecisionTreeNode * const aNode = (DecisionTreeNode *)pTreeTable->getArray();
    size_t nodeId                  = 0;
    if (parentId == noParent)
    {
        aNode[0].featureIndex           = featureIndex;
        aNode[0].defaultLeft            = defaultLeft;
        aNode[0].leftIndexOrClass       = 0;
        aNode[0].featureValueOrResponse = featureValue;
        aNode[0].cover                  = cover;
        nodeId                          = 0;
    }
    else if (aNode[parentId].featureIndex < 0)
    {
        return services::Status(services::ErrorID::ErrorIncorrectParameter);
    }
    else
    {
        /*if not leaf, and parent has child already*/
        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 1))
        {
            const size_t reservedId = aNode[parentId].leftIndexOrClass + 1;
            nodeId                  = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex           = featureIndex;
                aNode[nodeId].defaultLeft            = defaultLeft;
                aNode[nodeId].leftIndexOrClass       = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
                aNode[nodeId].cover                  = cover;
            }
        }
        if ((aNode[parentId].leftIndexOrClass > 0) && (position == 0))
        {
            const size_t reservedId = aNode[parentId].leftIndexOrClass;
            nodeId                  = reservedId;
            if (aNode[reservedId].featureIndex == __NODE_RESERVED_ID)
            {
                aNode[nodeId].featureIndex           = featureIndex;
                aNode[nodeId].defaultLeft            = defaultLeft;
                aNode[nodeId].leftIndexOrClass       = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
                aNode[nodeId].cover                  = cover;
            }
        }
        if ((aNode[parentId].leftIndexOrClass == 0) && (position == 0))
        {
            size_t i;
            for (i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    nodeId = i;
                    break;
                }
            }
            /* no space left */
            if (i == nRows)
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
            aNode[nodeId].featureIndex           = featureIndex;
            aNode[nodeId].defaultLeft            = defaultLeft;
            aNode[nodeId].leftIndexOrClass       = 0;
            aNode[nodeId].featureValueOrResponse = featureValue;
            aNode[nodeId].cover                  = cover;
            aNode[parentId].leftIndexOrClass     = nodeId;
            if (((nodeId + 1) < nRows) && (aNode[nodeId + 1].featureIndex == __NODE_FREE_ID))
            {
                aNode[nodeId + 1].featureIndex = __NODE_RESERVED_ID;
            }
            else
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
        }
        if ((aNode[parentId].leftIndexOrClass == 0) && (position == 1))
        {
            size_t leftEmptyId = 0;
            size_t i;
            for (i = parentId + 1; i < nRows; i++)
            {
                if (aNode[i].featureIndex == __NODE_FREE_ID)
                {
                    leftEmptyId = i;
                    break;
                }
            }
            /*if no free nodes leftBound is not initialized and no space left*/
            if (i == nRows)
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
            aNode[leftEmptyId].featureIndex  = __NODE_RESERVED_ID;
            aNode[parentId].leftIndexOrClass = leftEmptyId;
            nodeId                           = leftEmptyId + 1;
            if (nodeId < nRows)
            {
                aNode[nodeId].featureIndex           = featureIndex;
                aNode[nodeId].defaultLeft            = defaultLeft;
                aNode[nodeId].leftIndexOrClass       = 0;
                aNode[nodeId].featureValueOrResponse = featureValue;
                aNode[nodeId].cover                  = cover;
            }
            else
            {
                return services::Status(services::ErrorID::ErrorIncorrectParameter);
            }
        }
    }
    res = nodeId;
    return s;
}

} // namespace internal
} // namespace dtrees
} // namespace algorithms
} // namespace daal
