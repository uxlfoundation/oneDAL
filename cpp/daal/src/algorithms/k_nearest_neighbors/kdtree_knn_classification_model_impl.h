/* file: kdtree_knn_classification_model_impl.h */
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
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_MODEL_IMPL_
#define __KDTREE_KNN_CLASSIFICATION_MODEL_IMPL_

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace interface1
{
struct KDTreeNode
{
    size_t dimension;
    size_t leftIndex;
    size_t rightIndex;
    double cutPoint;
};

class KDTreeTable : public data_management::AOSNumericTable
{
public:
    KDTreeTable(size_t rowCount, services::Status & st);
    KDTreeTable(services::Status & st);
};
typedef services::SharedPtr<KDTreeTable> KDTreeTablePtr;
typedef services::SharedPtr<const KDTreeTable> KDTreeTableConstPtr;

class Model::ModelImpl
{
public:
    /**
     * Empty constructor for deserialization
     */
    ModelImpl(size_t nFeatures = 0) : _nFeatures(nFeatures), _kdTreeTable(), _rootNodeIndex(0), _lastNodeIndex(0), _data(), _labels(), _indices() {}

    /**
     * Returns the KD-tree table
     * \return KD-tree table
     */
    KDTreeTablePtr getKDTreeTable() { return _kdTreeTable; }

    /**
     * Returns the KD-tree table
     * \return KD-tree table
     */
    KDTreeTableConstPtr getKDTreeTable() const { return _kdTreeTable; }

    /**
     * Sets a KD-tree table
     * \param[in]  value  KD-tree table
     */
    void setKDTreeTable(const KDTreeTablePtr & value) { _kdTreeTable = value; }

    /**
     * Returns the index of KD-tree root node
     * \return Index of KD-tree root node
     */
    size_t getRootNodeIndex() const { return _rootNodeIndex; }

    /**
     * Sets a index of KD-tree root node
     * \param[in]  value  Index of KD-tree root node
     */
    void setRootNodeIndex(size_t value) { _rootNodeIndex = value; }

    /**
     * Returns the index of first part KD-tree last node
     * \return Index of first part KD-tree last node
     */
    size_t getLastNodeIndex() const { return _lastNodeIndex; }

    /**
    *  Sets a index of first part KD-tree last node
    *  \param[in]  value  Index of first part KD-tree last node
    */
    void setLastNodeIndex(size_t value) { _lastNodeIndex = value; }

    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTableConstPtr getData() const { return _data; }

    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTablePtr getData() { return _data; }

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->set(_nFeatures);
        arch->set(_rootNodeIndex);
        arch->set(_lastNodeIndex);
        arch->setSharedPtrObj(_kdTreeTable);
        arch->setSharedPtrObj(_data);
        arch->setSharedPtrObj(_labels);
        arch->setSharedPtrObj(_indices);

        return services::Status();
    }

    /**
     * Sets a training data
     * \param[in]  value  Training data
     * \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input data.
     */
    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setData(const data_management::NumericTablePtr & value, bool copy)
    {
        int result = 0;
        if (!copy)
        {
            _data = value;
        }
        else
        {
            data_management::SOANumericTablePtr tbl(
                new data_management::SOANumericTable(value->getNumberOfColumns(), value->getNumberOfRows(), data_management::DictionaryIface::equal));
            DAAL_CHECK_MALLOC(tbl.get())
            tbl->getDictionary()->setAllFeatures<algorithmFPType>(); // Just to set type of all features. Also, no way to use featuresEqual flag.
            tbl->resize(value->getNumberOfRows());

            tbl->allocateDataMemory();
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            tbl->getBlockOfRows(0, tbl->getNumberOfRows(), data_management::writeOnly, destBD);
            value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            result = services::internal::daal_memcpy_s(
                destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType), srcBD.getBlockPtr(),
                srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            tbl->releaseBlockOfRows(destBD);
            value->releaseBlockOfRows(srcBD);
            _data = tbl;
        }
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    /**
     * Returns training labels
     * \return Training labels
     */
    data_management::NumericTableConstPtr getLabels() const { return _labels; }

    /**
     * Returns training labels
     * \return Training labels
     */
    data_management::NumericTablePtr getLabels() { return _labels; }

    /**
     * Sets a training data
     * \param[in]  value  Training labels
     * \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input labels.
     */
    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setLabels(const data_management::NumericTablePtr & value, bool copy)
    {
        int result = 0;
        if (!copy)
        {
            _labels = value;
        }
        else
        {
            data_management::SOANumericTablePtr tbl(new data_management::SOANumericTable(value->getNumberOfColumns(), value->getNumberOfRows()));
            DAAL_CHECK_MALLOC(tbl.get())
            tbl->setArray(static_cast<algorithmFPType *>(0), 0);                    // Just to create the dictionary.
            tbl->getDictionary()->setNumberOfFeatures(value->getNumberOfColumns()); // Sadly, setArray() hides number of features from the dictionary.
            data_management::NumericTableFeature temp;
            temp.setType<algorithmFPType>();
            tbl->getDictionary()->setAllFeatures(temp); // Just to set type of all features. Also, no way to use featuresEqual flag.
            tbl->allocateDataMemory();
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            tbl->getBlockOfRows(0, tbl->getNumberOfRows(), data_management::writeOnly, destBD);
            value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            result = services::internal::daal_memcpy_s(
                destBD.getBlockPtr(), destBD.getNumberOfColumns() * destBD.getNumberOfRows() * sizeof(algorithmFPType), srcBD.getBlockPtr(),
                srcBD.getNumberOfColumns() * srcBD.getNumberOfRows() * sizeof(algorithmFPType));
            tbl->releaseBlockOfRows(destBD);
            value->releaseBlockOfRows(srcBD);
            _labels = tbl;
        }
        return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
    }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const { return _nFeatures; }

    /**
     * Returns training data original indices
     * \return Training data original indices
     */
    data_management::NumericTableConstPtr getIndices() const { return _indices; }

    /**
     * Returns training data original indices
     * \return Training data original indices
     */
    data_management::NumericTablePtr getIndices() { return _indices; }

    /**
     * Sets a training data original indices
     * \param[in]  value  Training data
     */
    DAAL_FORCEINLINE services::Status resetIndices(size_t nIndices)
    {
        typedef data_management::HomogenNumericTable<size_t> IndicesNT;

        services::Status status;

        _indices = IndicesNT::create(1, nIndices, data_management::NumericTableIface::doAllocate, &status);

        if (status.ok())
        {
            const auto ptr = static_cast<IndicesNT *>(_indices.get())->getArray();
            for (size_t i = 0; i < nIndices; ++i)
            {
                ptr[i] = i;
            }
        }

        return status;
    }

private:
    size_t _nFeatures;
    KDTreeTablePtr _kdTreeTable;
    size_t _rootNodeIndex;
    size_t _lastNodeIndex;
    data_management::NumericTablePtr _data;
    data_management::NumericTablePtr _labels;
    data_management::NumericTablePtr _indices;
};

} // namespace interface1

using interface1::KDTreeTable;
using interface1::KDTreeTablePtr;
using interface1::KDTreeTableConstPtr;
using interface1::KDTreeNode;

} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
