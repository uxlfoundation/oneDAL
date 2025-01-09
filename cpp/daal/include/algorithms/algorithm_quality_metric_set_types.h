/* file: algorithm_quality_metric_set_types.h */
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
//  Interface for the quality metric set.
//--
*/

#ifndef __ALGORITHM_QUALITY_METRIC_SET_TYPES_H__
#define __ALGORITHM_QUALITY_METRIC_SET_TYPES_H__

#include "data_management/data/input_collection.h"
#include "algorithms/algorithm_quality_metric_batch.h"

namespace daal
{
namespace algorithms
{
namespace quality_metric_set
{
/**
 * \brief Contains version 1.0 of the Intel(R) oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__INPUTALGORITHMSCOLLECTION"></a>
 * \brief Class that implements functionality of the collection of quality metrics algorithms
 */
class DAAL_EXPORT InputAlgorithmsCollection
{
public:
    /**
     * Constructs the collection of quality metrics algorithms
     * \param[in] n     Number of elements in the collection
     */
    InputAlgorithmsCollection(size_t n = 0);

    virtual ~InputAlgorithmsCollection();

    InputAlgorithmsCollection(InputAlgorithmsCollection & other) = delete;

    InputAlgorithmsCollection & operator=(InputAlgorithmsCollection & other) = delete;

    /**
     * Returns a reference to SharedPtr for a stored object with a given key if an object with such key is registered
     * \param[in] k     Key value
     * \return Reference to SharedPtr of the quality_metric::Batch type
     */
    const services::SharedPtr<quality_metric::Batch> & operator[](size_t k) const;

    /**
     * Returns a reference to SharedPtr for a stored object with a given key if an object with such key is registered.
     * Otherwise, creates an empty SharedPtr and stores it under the requested key and returns a reference for this value
     * \param[in] k     Key value
     * \return Reference to SharedPtr of the quality_metric::Batch type
     */
    services::SharedPtr<quality_metric::Batch> & operator[](size_t k);

    /**
     *  Returns the number of stored elements
     *  \return number of stored elements
     */
    size_t size() const;

    /**
     * Removes all elements from the container
     */
    void clear();

    /**
     *  Returns a reference to SharedPtr for the stored key with a given index
     *  \param[in]  idx  Index of the requested key
     *  \return Reference to SharedPtr of the size_t type
     */
    size_t getKeyByIndex(int idx);

protected:
    services::Collection<services::SharedPtr<quality_metric::Batch> > _qualityMetrics;
    services::Collection<size_t> _keys;
    services::SharedPtr<quality_metric::Batch> * nullPtr;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__INPUTDATACOLLECTION"></a>
 * \brief Class that implements functionality of the collection of input objects of the quality metrics algorithm
 */
class DAAL_EXPORT InputDataCollection : public data_management::KeyValueInputCollection
{
public:
    InputDataCollection();

    /**
     * Adds an element with a key to the collection
     * \param[in] k     Key value
     * \param[in] ptr   Shared pointer to the element
     */
    void add(size_t k, const algorithms::InputPtr & ptr);

    /**
     * Returns the element that matches the key
     * \param[in] key     Key value
     * \return Shared pointer to the element
     */
    algorithms::InputPtr getInput(size_t key) const;

    virtual ~InputDataCollection() {}
};
typedef services::SharedPtr<InputDataCollection> InputDataCollectionPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUALITY_METRIC_SET__RESULTCOLLECTION"></a>
 * \brief Class that implements functionality of the collection of result objects of the quality metrics algorithm
 */
class DAAL_EXPORT ResultCollection : public data_management::KeyValueDataCollection
{
public:
    ResultCollection();

    void add(size_t key, const algorithms::ResultPtr & ptr);

    algorithms::ResultPtr getResult(size_t key) const;

    virtual ~ResultCollection() {}
};
typedef services::SharedPtr<ResultCollection> ResultCollectionPtr;
/** @} */
} // namespace interface1
using interface1::InputAlgorithmsCollection;
using interface1::InputDataCollection;
using interface1::InputDataCollectionPtr;
using interface1::ResultCollection;
using interface1::ResultCollectionPtr;

} // namespace quality_metric_set
} // namespace algorithms
} // namespace daal
#endif
