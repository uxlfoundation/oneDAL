/* file: implicit_als_train_distributed.cpp */
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
//  Implementation of implicit als algorithm and types methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"
#include "src/services/service_data_utils.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(PartialModel, SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID);
}

namespace training
{
namespace interface1
{
DistributedInput<step1Local>::DistributedInput() : daal::algorithms::Input(lastPartialModelInputId + 1) {}

/**
 * Returns an input object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
PartialModelPtr DistributedInput<step1Local>::get(PartialModelInputId id) const
{
    return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step1Local>::set(PartialModelInputId id, const PartialModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the parameters and input objects for the implicit ALS training algorithm
 * in the first step of the distributed processing mode
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step1Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * alsParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = alsParameter->nFactors;

    PartialModelPtr model = get(partialModel);
    DAAL_CHECK(model, ErrorNullPartialModel);
    services::Status s;
    int unexpectedLayouts = (int)packed_mask;
    s |= checkNumericTable(model->getFactors().get(), factorsStr(), unexpectedLayouts, 0, nFactors, 0);
    if (!s) return s;

    unexpectedLayouts = (int)NumericTableIface::csrArray;
    s |= checkNumericTable(model->getIndices().get(), indicesStr(), unexpectedLayouts, 0, 1, model->getFactors()->getNumberOfRows());
    return s;
}

/** Default constructor */
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep1, SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID);
DistributedPartialResultStep1::DistributedPartialResultStep1() : daal::algorithms::PartialResult(lastDistributedPartialResultStep1Id + 1) {}

/**
 * Returns a partial result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the partial result
 * \return          Value that corresponds to the given identifier
 */
data_management::NumericTablePtr DistributedPartialResultStep1::get(DistributedPartialResultStep1Id id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the new partial result object
 */
void DistributedPartialResultStep1::set(DistributedPartialResultStep1Id id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks a partial result of the implicit ALS algorithm
 * \param[in] input     %Input object for the algorithm
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method
 */
services::Status DistributedPartialResultStep1::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                      int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * alsParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = alsParameter->nFactors;
    int unexpectedLayout           = (int)NumericTableIface::csrArray;
    services::Status s;
    s |= checkNumericTable(get(outputOfStep1ForStep2).get(), crossProductStr(), unexpectedLayout, 0, nFactors, nFactors);
    if (!s) return s;

    const DistributedInput<step1Local> * alsInput = static_cast<const DistributedInput<step1Local> *>(input);
    DAAL_CHECK(alsInput->get(partialModel), ErrorNullPartialModel);
    return s;
}

/** Default constructor */
DistributedInput<step2Master>::DistributedInput() : daal::algorithms::Input(lastMasterInputId + 1)
{
    Argument::set(inputOfStep2FromStep1, data_management::DataCollectionPtr(new data_management::DataCollection()));
}

/**
 * Returns an input object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::DataCollectionPtr DistributedInput<step2Master>::get(MasterInputId id) const
{
    return services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step2Master>::set(MasterInputId id, const data_management::DataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds an input object for the implicit ALS training algorithm in the second step
 * of the distributed processing mode
 * \param[in] id            Identifier of the input object
 * \param[in] partialResult Pointer to the partial result obtained in the previous step of the distributed processing mode
 */
void DistributedInput<step2Master>::add(MasterInputId id, const DistributedPartialResultStep1Ptr & partialResult)
{
    data_management::DataCollectionPtr collection =
        services::staticPointerCast<data_management::DataCollection, data_management::SerializationIface>(Argument::get(id));
    if (!collection)
    {
        return;
    }
    if (!partialResult)
    {
        return;
    }
    collection->push_back(partialResult->get(training::outputOfStep1ForStep2));
}

/**
 * Checks the parameters and input objects for the implicit ALS training algorithm in the second step
 * of the distributed processing mode
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * alsParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = alsParameter->nFactors;

    DataCollectionPtr collectionPtr = get(inputOfStep2FromStep1);
    DAAL_CHECK(collectionPtr, ErrorNullInputDataCollection);

    size_t nBlocks = collectionPtr->size();
    DAAL_CHECK(collectionPtr->size(), ErrorIncorrectNumberOfInputNumericTables);

    services::Status s;
    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_CHECK_EX((*collectionPtr)[i], ErrorNullNumericTable, ArgumentName, inputOfStep2FromStep1Str());
        NumericTablePtr nt = NumericTable::cast((*collectionPtr)[i]);
        DAAL_CHECK_EX(nt, ErrorIncorrectElementInNumericTableCollection, ArgumentName, inputOfStep2FromStep1Str());
        int unexpectedLayouts = (int)packed_mask;
        s |= checkNumericTable(nt.get(), inputOfStep2FromStep1Str(), unexpectedLayouts, 0, nFactors, nFactors);
        if (!s) return s;
    }
    return s;
}

/** Default constructor */
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep2, SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID);
DistributedPartialResultStep2::DistributedPartialResultStep2() : daal::algorithms::PartialResult(lastDistributedPartialResultStep2Id + 1) {}

/**
 * Returns a partial result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the partial result
 * \return          Value that corresponds to the given identifier
 */
data_management::NumericTablePtr DistributedPartialResultStep2::get(DistributedPartialResultStep2Id id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the new partial result object
 */
void DistributedPartialResultStep2::set(DistributedPartialResultStep2Id id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the structure of input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
services::Status DistributedPartialResultStep2::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                      int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = algParameter->nFactors;
    int unexpectedLayouts          = (int)packed_mask;
    services::Status s;
    s |= checkNumericTable(get(outputOfStep2ForStep4).get(), outputOfStep2ForStep4Str(), unexpectedLayouts, 0, nFactors, nFactors);
    return s;
}
/** Default constructor */
DistributedInput<step3Local>::DistributedInput() : daal::algorithms::Input(lastStep3LocalNumericTableInputId + 1) {}

/**
 * Returns an input partial model object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
PartialModelPtr DistributedInput<step3Local>::get(PartialModelInputId id) const
{
    return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns an input key-value data collection object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::KeyValueDataCollectionPtr DistributedInput<step3Local>::get(Step3LocalCollectionInputId id) const
{
    return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns an input numeric table object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr DistributedInput<step3Local>::get(Step3LocalNumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input partial model object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step3Local>::set(PartialModelInputId id, const PartialModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input key-value data collection object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step3Local>::set(Step3LocalCollectionInputId id, const data_management::KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input numeric table object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step3Local>::set(Step3LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of blocks of data used in distributed computations
 * \return Number of blocks of data
 */
size_t DistributedInput<step3Local>::getNumberOfBlocks() const
{
    data_management::KeyValueDataCollectionPtr outBlocksCollection = get(partialModelBlocksToNode);
    if (outBlocksCollection) return outBlocksCollection->size();
    return 0;
}

/**
 * Returns the index of the starting row of the input partial model
 * \return Index of the starting row of the input partial model
 */
size_t DistributedInput<step3Local>::getOffset() const
{
    data_management::NumericTablePtr offsetTable = get(offset);
    if (offsetTable)
    {
        data_management::BlockDescriptor<int> block;
        offsetTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        int iOffset = (block.getBlockPtr())[0];
        DAAL_ASSERT(iOffset >= 0)
        size_t offset = (size_t)iOffset;
        offsetTable->releaseBlockOfRows(block);
        return offset;
    }
    return 0;
}

/**
 * Returns the numeric table that contains the indices of factors that should be transferred to a specified node
 * \param[in] key Index of the node
 * \return Numeric table that contains the indices of factors that should be transferred to a specified node
 */
data_management::NumericTablePtr DistributedInput<step3Local>::getOutBlockIndices(size_t key) const
{
    data_management::NumericTablePtr outBlockIndices;
    data_management::KeyValueDataCollectionPtr outBlocksCollection = get(partialModelBlocksToNode);
    if (outBlocksCollection)
        outBlockIndices =
            services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*outBlocksCollection)[key]);
    return outBlockIndices;
}

/**
 * Checks the parameters and input objects of the implicit ALS training algorithm in the first step of
 * the distributed processing mode
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step3Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = algParameter->nFactors;

    /* Check offset numeric table */
    int unexpectedLayoutsCSR    = (int)NumericTableIface::csrArray;
    int unexpectedLayoutsPacked = (int)packed_mask;
    services::Status s;
    s |= checkNumericTable(get(offset).get(), offsetStr(), unexpectedLayoutsCSR, 0, 1, 1);
    if (!s) return s;

    /* Check input partial model */
    PartialModelPtr model = get(partialModel);
    DAAL_CHECK(model, ErrorNullPartialModel);
    s |= checkNumericTable(model->getIndices().get(), indicesStr(), unexpectedLayoutsCSR, 0, 1, 0);
    if (!s) return s;
    s |= checkNumericTable(model->getFactors().get(), factorsStr(), unexpectedLayoutsPacked, 0, nFactors, model->getIndices()->getNumberOfRows());
    if (!s) return s;

    /* Check input collection */
    KeyValueDataCollectionPtr collection = get(partialModelBlocksToNode);
    DAAL_CHECK(collection, ErrorNullInputDataCollection);
    size_t nBlocks = collection->size();

    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialModelBlocksToNodeStr());

    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_ASSERT(i <= services::internal::MaxVal<int>::get())
        DAAL_CHECK_EX(collection->getValueByIndex((int)i), ErrorNullNumericTable, ArgumentName, blockIndicesStr());
        NumericTablePtr blockIndices = NumericTable::cast(collection->getValueByIndex((int)i));
        DAAL_CHECK_EX(blockIndices, ErrorIncorrectElementInNumericTableCollection, ArgumentName, blockIndicesStr());
        s |= checkNumericTable(blockIndices.get(), blockIndicesStr(), unexpectedLayoutsCSR, 0, 1);
        if (!s) return s;
        DAAL_CHECK_EX(model->getIndices()->getNumberOfRows() >= blockIndices->getNumberOfRows(), ErrorIncorrectNumberOfRows, ArgumentName,
                      blockIndicesStr());
    }
    return s;
}

/** Default constructor */
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep3, SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID);
DistributedPartialResultStep3::DistributedPartialResultStep3() : daal::algorithms::PartialResult(lastDistributedPartialResultStep3Id + 1) {}

/**
 * Returns a partial result of the implicit ALS training algorithm
 *
 * \param[in] id    Identifier of the partial result
 * \return          Value that corresponds to the given identifier
 */
data_management::KeyValueDataCollectionPtr DistributedPartialResultStep3::get(DistributedPartialResultStep3Id id) const
{
    return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns a partial model obtained with the compute() method of the implicit ALS algorithm in the third step of the
 * distributed processing mode
 *
 * \param[in] id    Identifier of the partial result
 * \param[in] key   Index of the partial model in the key-value data collection
 * \return          Pointer to the partial model object
 */
PartialModelPtr DistributedPartialResultStep3::get(DistributedPartialResultStep3Id id, size_t key) const
{
    PartialModelPtr model;
    data_management::KeyValueDataCollectionPtr collection = get(id);
    if (collection) model = services::staticPointerCast<PartialModel, data_management::SerializationIface>((*collection)[key]);
    return model;
}

/**
 * Sets a partial result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the input object
 */
void DistributedPartialResultStep3::set(DistributedPartialResultStep3Id id, const data_management::KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the structure of input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
services::Status DistributedPartialResultStep3::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                      int method) const
{
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * algParameter                = static_cast<const Parameter *>(parameter);
    const DistributedInput<step3Local> * algInput = static_cast<const DistributedInput<step3Local> *>(input);
    size_t nFactors                               = algParameter->nFactors;

    KeyValueDataCollectionPtr collection = get(outputOfStep3ForStep4);
    DAAL_CHECK(collection, ErrorNullOutputDataCollection);
    size_t nBlocks = collection->size();
    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialModelBlocksToNodeStr());

    services::Status s;
    int unexpectedLayouts = (int)packed_mask;
    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_ASSERT(i <= services::internal::MaxVal<int>::get())
        DAAL_CHECK_EX(collection->getValueByIndex((int)i), ErrorNullPartialModel, ArgumentName, outputOfStep3ForStep4Str());
        PartialModelPtr model = PartialModel::cast(collection->getValueByIndex((int)i));
        DAAL_CHECK_EX(model, ErrorIncorrectElementInPartialResultCollection, ArgumentName, outputOfStep3ForStep4Str());
        NumericTablePtr nt = algInput->getOutBlockIndices(i);
        DAAL_CHECK_EX(nt, ErrorNullNumericTable, ArgumentName, outBlockIndicesStr());
        size_t nRows = nt->getNumberOfRows();
        s |= checkNumericTable(model->getFactors().get(), factorsStr(), unexpectedLayouts, 0, nFactors, nRows);
        if (!s) return s;
        s |= checkNumericTable(model->getIndices().get(), indicesStr(), unexpectedLayouts, 0, 1, nRows);
        if (!s) return s;
    }
    return s;
}
/** Default constructor */
DistributedInput<step4Local>::DistributedInput() : daal::algorithms::Input(lastStep4LocalNumericTableInputId + 1) {}

/**
 * Returns an input key-value data collection object for the implicit ALS training algorithm
 *
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier.
 *                  A key-value data collection contains partial models consisting of user factors/item factors
 *                  computed in the third step of the distributed processing mode
 */
data_management::KeyValueDataCollectionPtr DistributedInput<step4Local>::get(Step4LocalPartialModelsInputId id) const
{
    return services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns an input numeric table object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr DistributedInput<step4Local>::get(Step4LocalNumericTableInputId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input key-value data collection object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step4Local>::set(Step4LocalPartialModelsInputId id, const data_management::KeyValueDataCollectionPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Sets an input numeric table object for the implicit ALS training algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the new input object value
 */
void DistributedInput<step4Local>::set(Step4LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns the number of rows in the partial matrix of users factors/items factors
 * \return Number of rows in the partial matrix of factors
 */
size_t DistributedInput<step4Local>::getNumberOfRows() const
{
    data_management::NumericTablePtr dataTable = get(partialData);
    if (dataTable) return dataTable->getNumberOfRows();
    return 0;
}

/**
 * Checks the parameters and input objects for the implicit ALS training algorithm in the first step of
 * the distributed processing mode
 * \param[in] parameter %Parameter of the algorithm
 * \param[in] method    Computation method of the algorithm
 */
services::Status DistributedInput<step4Local>::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = algParameter->nFactors;

    /* Check input numeric tables */
    NumericTablePtr dataTable = get(partialData);
    if (!dataTable) return services::Status(services::ErrorNullNumericTable);

    NumericTablePtr crossProduct = get(inputOfStep4FromStep2);

    int expectedLayout = (int)NumericTableIface::csrArray;
    s |= checkNumericTable(get(partialData).get(), partialDataStr(), 0, expectedLayout);
    if (!s) return s;
    int unexpectedLayoutsPacked = (int)packed_mask;
    int unexpectedLayoutsCSR    = (int)packed_mask;
    s |= checkNumericTable(crossProduct.get(), crossProductStr(), unexpectedLayoutsPacked, 0, nFactors, nFactors);
    if (!s) return s;
    /* Check input data collection */
    KeyValueDataCollectionPtr collection = get(partialModels);
    DAAL_CHECK(collection, ErrorNullInputDataCollection);
    size_t nBlocks = collection->size();
    DAAL_CHECK_EX(nBlocks > 0, ErrorIncorrectNumberOfInputNumericTables, ArgumentName, partialModelBlocksToNodeStr());
    for (size_t i = 0; i < nBlocks; i++)
    {
        DAAL_ASSERT(i <= services::internal::MaxVal<int>::get())
        DAAL_CHECK_EX(collection->getValueByIndex((int)i), ErrorNullPartialModel, ArgumentName, partialModelsStr());
        PartialModelPtr model = PartialModel::cast(collection->getValueByIndex((int)i));
        DAAL_CHECK_EX(model, ErrorIncorrectElementInPartialResultCollection, ArgumentName, partialModelsStr());
        s |= checkNumericTable(model->getFactors().get(), factorsStr(), unexpectedLayoutsPacked, 0, nFactors);
        if (!s) return s;
        size_t nRows = model->getFactors()->getNumberOfRows();
        s |= checkNumericTable(model->getIndices().get(), indicesStr(), unexpectedLayoutsCSR, 0, 1, nRows);
        if (!s) return s;
    }
    return s;
}

/** Default constructor */
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep4, SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID);
DistributedPartialResultStep4::DistributedPartialResultStep4() : daal::algorithms::PartialResult(lastDistributedPartialResultStep4Id + 1) {}

/**
 * Returns a partial result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the partial result
 * \return          Value that corresponds to the given identifier
 */
PartialModelPtr DistributedPartialResultStep4::get(DistributedPartialResultStep4Id id) const
{
    return services::staticPointerCast<PartialModel, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets a partial result of the implicit ALS training algorithm
 * \param[in] id    Identifier of the partial result
 * \param[in] ptr   Pointer to the partial result
 */
void DistributedPartialResultStep4::set(DistributedPartialResultStep4Id id, const PartialModelPtr & ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks a partial result of the implicit ALS training algorithm
 * \param[in] input     Pointer to the structure of input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
services::Status DistributedPartialResultStep4::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                      int method) const
{
    services::Status s;
    DAAL_CHECK(method == fastCSR, ErrorMethodNotSupported);

    const Parameter * algParameter = static_cast<const Parameter *>(parameter);
    size_t nFactors                = algParameter->nFactors;

    PartialModelPtr model = get(outputOfStep4);
    DAAL_CHECK(model, ErrorNullPartialModel);

    int unexpectedLayouts = (int)packed_mask;
    s |= checkNumericTable(model->getFactors().get(), factorsStr(), unexpectedLayouts, 0, nFactors);
    if (!s) return s;

    size_t nRows      = model->getFactors()->getNumberOfRows();
    unexpectedLayouts = (int)NumericTableIface::csrArray;
    s |= checkNumericTable(model->getIndices().get(), indicesStr(), unexpectedLayouts, 0, 1, nRows);
    return s;
}

} // namespace interface1
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
