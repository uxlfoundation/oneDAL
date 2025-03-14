/* file: kdtree_knn_classification_training_types.h */
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
//  Implementation of the k-Nearest Neighbor (kNN) algorithm interface
//--
*/

#ifndef __KNN_CLASSIFICATION_TRAINING_TYPES_H__
#define __KNN_CLASSIFICATION_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/data_serialize.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_model.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes of the KD-tree based kNN algorithm
 */
namespace kdtree_knn_classification
{
/**
 * @defgroup kdtree_knn_classification_training Training
 * \copydoc daal::algorithms::kdtree_knn_classification::training
 * @ingroup kdtree_knn_classification
 * @{
 */
/**
 * \brief Contains a class for KD-tree based kNN model-based training
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__METHOD"></a>
 * \brief Computation methods for KD-tree based kNN model-based training
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__INPUT"></a>
 * \brief %Input objects for KD-tree based kNN model-based training
 */
class DAAL_EXPORT Input : public classifier::training::Input
{
public:
    Input() : classifier::training::Input() {}
    Input(const Input & other) : classifier::training::Input(other) {}

    /**
     * Checks the correctness of the input object
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter * parameter) const;
};
typedef services::SharedPtr<Input> InputPtr;
typedef services::SharedPtr<const Input> InputConstPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KDTREE_KNN_CLASSIFICATION__TRAINING__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of KD-tree based kNN model-based training
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    /**
     * Returns the result of KD-tree based kNN model-based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    daal::algorithms::kdtree_knn_classification::interface1::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Allocates memory to store the result of KD-tree based kNN model-based training
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of KD-tree based kNN model-based training
     * \param[in] method Computation method for the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const Parameter * parameter, int method);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return classifier::training::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1

using interface1::Input;
using interface1::InputPtr;
using interface1::InputConstPtr;
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
