/* file: dbscan_types.h */
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
//  Implementation of the DBSCAN algorithm interface.
//--
*/

#ifndef __DBSCAN_TYPES_H__
#define __DBSCAN_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup dbscan DBSCAN
 * \copydoc daal::algorithms::dbscan
 * @ingroup analysis
 * @defgroup dbscan_compute Computation
 * \copydoc daal::algorithms::dbscan
 * @ingroup dbscan
 * @{
 */
/** \brief Contains classes of the DBSCAN algorithm */
namespace dbscan
{
const int noise     = -1;
const int undefined = -2;

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__METHOD"></a>
 * Available methods of the DBSCAN algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTANCETYPE"></a>
 * Supported distance types
 */
enum DistanceType
{
    euclidean, /*!< Euclidean distance */
    lastDistanceType = euclidean
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__INPUTID"></a>
 * \brief Available identifiers of input objects for the DBSCAN algorithm
 */
enum InputId
{
    data,    /*!< %Input data table */
    weights, /*!< %Input weights of observations */
    lastInputId = weights
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__RESULTTOCOMPUTEID"></a>
 * Available identifiers to specify the result to compute
 */
enum ResultToComputeId
{
    computeCoreIndices      = 0x00000001ULL, /*!< Compute table containing indices of core observations */
    computeCoreObservations = 0x00000002ULL  /*!< Compute table containing core observations */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__RESULTID"></a>
 * \brief Available identifiers of results of the DBSCAN algorithm
 */
enum ResultId
{
    assignments,      /*!< Table containing assignments of observations to clusters */
    nClusters,        /*!< Table containing number of clusters */
    coreIndices,      /*!< Table containing indices of core observations */
    coreObservations, /*!< Table containing core observations */
    lastResultId = coreObservations
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the distributed processing mode
 */
enum LocalCollectionInputId
{
    partialData,    /*!< Collection of input data tables that contains observations */
    partialWeights, /*!< Collection of input data tables that contains weights of observations */
    lastLocalCollectionInputId = partialWeights
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP1LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for the DBSCAN algorithm in the first step
 * of the distributed processing mode
 */
enum Step1LocalNumericTableInputId
{
    step1Data, /*!< Input data table that contains observations */
    lastStep1LocalNumericTableInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP1ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the first step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep1Id
{
    partialOrder, /*!< Table containing information about observations:
                                                             identifier of initial block and index in initial block */
    lastDistributedPartialResultStep1Id = partialOrder
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP2ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the second step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep2Id
{
    boundingBox, /*!< Table containing bounding box of input observations:
                                                             first row contains minimum value of each feature,
                                                             second row contains maximum value of each feature. */
    lastDistributedPartialResultStep2Id = boundingBox
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP3LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the third step
 * of the distributed processing mode
 */
enum Step3LocalCollectionInputId
{
    step3PartialBoundingBoxes       = lastLocalCollectionInputId + 1, /*!< Collection of input tables containing bounind boxes */
    lastStep3LocalCollectionInputId = step3PartialBoundingBoxes
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP3ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the third step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep3Id
{
    split, /*!< Table containing information about split for current
                                                            iteration of geometric repartitioning */
    lastDistributedPartialResultStep3Id = split
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP4LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the fourth step
 * of the distributed processing mode
 */
enum Step4LocalCollectionInputId
{
    step4PartialSplits = lastLocalCollectionInputId + 1, /*!< Collection of input tables containing information about split
                                                                for current iteration of geometric repartitioning */
    step4PartialOrders,                                  /*!< Collection of input tables containing information about observations:
                                                                identifier of initial block and index in initial block */
    lastStep4LocalCollectionInputId = step4PartialOrders
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP4ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the fourth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep4Id
{
    partitionedData,          /*!< Collection of tables containing observations */
    partitionedWeights,       /*!< Collection of tables containing weights of observations */
    partitionedPartialOrders, /*!< Collection of tables containing information about observations:
                                                                         identifier of initial block and index in initial block */
    lastDistributedPartialResultStep4Id = partitionedPartialOrders
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP5LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the fifth step
 * of the distributed processing mode
 */
enum Step5LocalCollectionInputId
{
    step5PartialBoundingBoxes       = lastLocalCollectionInputId + 1, /*!< Collection of input tables containing bounding boxes */
    lastStep5LocalCollectionInputId = step5PartialBoundingBoxes
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP5ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the fifth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep5Id
{
    partitionedHaloData,        /*!< Collection of tables containing halo observations */
    partitionedHaloDataIndices, /*!< Collection of tables containing indices of halo observations */
    partitionedHaloWeights,     /*!< Collection of tables containing weights of halo observations */
    lastDistributedPartialResultStep5Id = partitionedHaloWeights
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP6LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the sixth step
 * of the distributed processing mode
 */
enum Step6LocalCollectionInputId
{
    haloData = lastLocalCollectionInputId + 1, /*!< Collection of input tables containing halo observations */
    haloDataIndices,                           /*!< Collection of input tables containing indices of halo observations*/
    haloWeights,                               /*!< Collection of input tables containing weights of halo observations*/
    haloBlocks,                                /*!< Collection of input tables containing identifiers of blocks for halo observations */
    lastStep6LocalCollectionInputId = haloBlocks
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP6NUMERICTABLEID"></a>
 * Available types of partial results of the DBSCAN algorithm in the sixth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep6NumericTableId
{
    step6ClusterStructure, /*!< Table containing information about current clustering state of observations */
    step6FinishedFlag,     /*!< Table containing the flag indicating that the clustering process is finished */
    step6NClusters,        /*!< Table containing the current number of clusters */
    lastDistributedPartialResultStep6NumericTableId = step6NClusters
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP6COLLECTIONID"></a>
 * Available types of partial results of the DBSCAN algorithm in the sixth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep6CollectionId
{
    step6Queries = lastDistributedPartialResultStep6NumericTableId + 1, /*!< Collection of tables containing clustering queries */
    lastDistributedPartialResultStep6CollectionId = step6Queries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP7MASTERCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the seventh step
 * of the distributed processing mode
 */
enum Step7MasterCollectionInputId
{
    partialFinishedFlags, /*!< Collection of input tables containing the flags
                                                                   indicating that the clustering process is finished */
    lastStep7MasterCollectionInputId = partialFinishedFlags
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP7ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the seventh step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep7Id
{
    finishedFlag, /*!< Table containing the flag indicating that the clustering process is finished */
    lastDistributedPartialResultStep7Id = finishedFlag
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP8LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for the DBSCAN algorithm in the eighth step
 * of the distributed processing mode
 */
enum Step8LocalNumericTableInputId
{
    step8InputClusterStructure, /*!<  Input table containing information about current clustering state of observations */
    step8InputNClusters,        /*!<  Input table containing the current number of clusters */
    lastStep8LocalNumericTableInputId = step8InputNClusters
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP8LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the eighth step
 * of the distributed processing mode
 */
enum Step8LocalCollectionInputId
{
    step8PartialQueries             = lastStep8LocalNumericTableInputId + 1, /*!<  Collection of input tables containing clustering queries */
    lastStep8LocalCollectionInputId = step8PartialQueries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP8NUMERICTABLEID"></a>
 * Available types of partial results of the DBSCAN algorithm in the eighth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep8NumericTableId
{
    step8ClusterStructure, /*!< Table containing information about current clustering state of observations */
    step8FinishedFlag,     /*!< Table containing the flag indicating that the clustering process is finished */
    step8NClusters,        /*!< Table containing the current number of clusters */
    lastDistributedPartialResultStep8NumericTableId = step8NClusters
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP8COLLECTIONID"></a>
 * Available types of partial results of the DBSCAN algorithm in the eighth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep8CollectionId
{
    step8Queries = lastDistributedPartialResultStep8NumericTableId + 1, /*!< Collection of tables containing clustering queries */
    lastDistributedPartialResultStep8CollectionId = step8Queries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP9MASTERCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the ninth step
 * of the distributed processing mode
 */
enum Step9MasterCollectionInputId
{
    partialNClusters, /*!< Collection of input tables containing the current number of clusters */
    lastStep9MasterCollectionInputId = partialNClusters
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDRESULTSTEP9ID"></a>
 * Available types of results of the DBSCAN algorithm in the ninth step
 * of the distributed processing mode
 */
enum DistributedResultStep9Id
{
    step9NClusters, /*!< Table containing the total number of clusters */
    lastDistributedResultStep9Id = step9NClusters
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP9ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the ninth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep9Id
{
    clusterOffsets, /*!< Collection of tables containing offsets for cluster numeration */
    lastDistributedPartialResultStep9Id = clusterOffsets
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP10LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data numeric table objects for the DBSCAN algorithm in the tenth step
 * of the distributed processing mode
 */
enum Step10LocalNumericTableInputId
{
    step10InputClusterStructure, /*!< Input table containing information about current clustering state of observations */
    step10ClusterOffset,         /*!< Input table containing the cluster numeration offset */
    lastStep10LocalNumericTableInputId = step10ClusterOffset
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP10NUMERICTABLEID"></a>
 * Available types of partial results of the DBSCAN algorithm in the tenth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep10NumericTableId
{
    step10ClusterStructure, /*!< Table containing information about current clustering state of observations */
    step10FinishedFlag,     /*!< Table containing the flag indicating that the cluster numerating process is finished */
    lastDistributedPartialResultStep10NumericTableId = step10FinishedFlag
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP10COLLECTIONID"></a>
 * Available types of partial results of the DBSCAN algorithm in the tenth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep10CollectionId
{
    step10Queries = lastDistributedPartialResultStep10NumericTableId + 1, /*!< Collection of tables containing cluster numerating queries */
    lastDistributedPartialResultStep10CollectionId = step10Queries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP11LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for the DBSCAN algorithm in the eleventh step
 * of the distributed processing mode
 */
enum Step11LocalNumericTableInputId
{
    step11InputClusterStructure, /*!< Input table containing information about current clustering state of observations */
    lastStep11LocalNumericTableInputId = step11InputClusterStructure
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP11LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the eleventh step
 * of the distributed processing mode
 */
enum Step11LocalCollectionInputId
{
    step11PartialQueries = lastStep11LocalNumericTableInputId + 1, /*!< Collection of input tables containing cluster numerating queries */
    lastStep11LocalCollectionInputId = step11PartialQueries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP11NUMERICTABLEID"></a>
 * Available types of partial results of the DBSCAN algorithm in the eleventh step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep11NumericTableId
{
    step11ClusterStructure, /*!< Table containing information about current clustering state of observations */
    step11FinishedFlag,     /*!< Table containing the flag indicating that the cluster numerating process is finished */
    lastDistributedPartialResultStep11NumericTableId = step11FinishedFlag
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP11COLLECTIONID"></a>
 * Available types of partial results of the DBSCAN algorithm in the eleventh step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep11CollectionId
{
    step11Queries = lastDistributedPartialResultStep11NumericTableId + 1, /*!< Collection of input tables containing cluster numerating queries */
    lastDistributedPartialResultStep11CollectionId = step11Queries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP12LOCALNUMERICTABLEINPUTID"></a>
 * Available identifiers of input data numeric table objects for the DBSCAN algorithm in the twelfth step
 * of the distributed processing mode
 */
enum Step12LocalNumericTableInputId
{
    step12InputClusterStructure, /*!< Input table containing information about current clustering state of observations */
    lastStep12LocalNumericTableInputId = step12InputClusterStructure
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP12LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the twelfth step
 * of the distributed processing mode
 */
enum Step12LocalCollectionInputId
{
    step12PartialOrders = lastStep12LocalNumericTableInputId + 1, /*!< Collection of input tables containing information about observations:
                                                                          identifier of initial block and index in initial block */
    lastStep12LocalCollectionInputId = step12PartialOrders
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP12ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the twelfth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep12Id
{
    assignmentQueries, /*!< Collection of tables containing cluster assigning queries */
    lastDistributedPartialResultStep12Id = assignmentQueries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__STEP12LOCALCOLLECTIONINPUTID"></a>
 * Available identifiers of input data collection objects for the DBSCAN algorithm in the thirteenth step
 * of the distributed processing mode
 */
enum Step13LocalCollectionInputId
{
    partialAssignmentQueries, /*!< Collection of input tables containing cluster assigning queries */
    lastStep13LocalCollectionInputId = partialAssignmentQueries
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDRESULTSTEP13ID"></a>
 * Available types of results of the DBSCAN algorithm in the thirteenth step
 * of the distributed processing mode
 */
enum DistributedResultStep13Id
{
    step13Assignments, /*!< Table containing assignments of observations to clusters */
    lastDistributedResultStep13Id = step13Assignments
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP13ID"></a>
 * Available types of partial results of the DBSCAN algorithm in the thirteenth step
 * of the distributed processing mode
 */
enum DistributedPartialResultStep13Id
{
    step13AssignmentQueries, /*!< Table containing assigning queries */
    lastDistributedPartialResultStep13Id = step13AssignmentQueries
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__DBSCAN__PARAMETER"></a>
 * \brief Parameters for the DBSCAN algorithm
 * \par Enumerations
 *      - \ref DistanceType Methods for distance computation
 *
 * \snippet dbscan/dbscan_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Constructs parameters of the DBSCAN algorithm
     */
    Parameter();

    /**
     *  Constructs parameters of the DBSCAN algorithm
     *  \param[in] _epsilon         Radius of neighborhood
     *  \param[in] _minObservations Minimal total weight of observations in neighborhood of core observation
     */
    Parameter(double _epsilon, size_t _minObservations);

    /**
     *  Constructs parameters of the DBSCAN algorithm by copying another parameters of the DBSCAN algorithm
     *  \param[in] other    Parameters of the DBSCAN algorithm
     */
    Parameter(const Parameter & other);

    double epsilon;               /*!< Radius of neighborhood */
    size_t minObservations;       /*!< Minimal total weight of observations in neighborhood of core observation */
    bool memorySavingMode;        /*!< If true then use memory saving (but slower) mode */
    DAAL_UINT64 resultsToCompute; /*!< 64 bit integer flag that indicates the results to compute */

    size_t blockIndex; /*!< Unique identifier of block initially passed for computation on the local node */
    size_t nBlocks;    /*!< Number of blocks initially passed for computation on all nodes */

    size_t leftBlocks;  /*!< Number of blocks that will process observations with value of selected
                                       split feature lesser than selected split value */
    size_t rightBlocks; /*!< Number of blocks that will process observations with value of selected
                                       split feature greater than selected split value */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__INPUT"></a>
 * \brief %Input objects for the DBSCAN algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other) : daal::algorithms::Input(other) {}

    virtual ~Input() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks input objects for the DBSCAN algorithm
     * \param[in] par     Algorithm parameter
     * \param[in] method  Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__RESULT"></a>
 * \brief Results obtained with the compute() method of the DBSCAN algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the DBSCAN algorithm
     * \param[in] input     Pointer to the structure of the input objects
     * \param[in] parameter Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns the result of the DBSCAN algorithm
     * \param[in] id   Result identifier
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the DBSCAN algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the result of the DBSCAN algorithm
     * \param[in] input   %Input objects for the algorithm
     * \param[in] par     Algorithm parameter
     * \param[in] method  Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT"></a>
 * \brief %Input objects for the DBSCAN algorithm in the distributed processing mode
 */
template <ComputeStep step>
class DistributedInput
{};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the first step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step1Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step1LocalNumericTableInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step1LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the first step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP1"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the first step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep1 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep1)
    /** Default constructor */
    DistributedPartialResultStep1();

    virtual ~DistributedPartialResultStep1() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep1Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep1Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep1> DistributedPartialResultStep1Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP2LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the second step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step2Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the second step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP2"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the second step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep2 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep2)
    /** Default constructor */
    DistributedPartialResultStep2();

    virtual ~DistributedPartialResultStep2() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep2Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep2Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep2> DistributedPartialResultStep2Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP3LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the third step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step3Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(LocalCollectionInputId id) const;

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step3LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step3LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step3LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the third step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the third step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep3 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep3)
    /** Default constructor */
    DistributedPartialResultStep3();

    virtual ~DistributedPartialResultStep3() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep3Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep3Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep3> DistributedPartialResultStep3Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP4LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the fourth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step4Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(LocalCollectionInputId id) const;

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step4LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step4LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step4LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the third step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP4"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the fourth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep4 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep4)
    /** Default constructor */
    DistributedPartialResultStep4();

    virtual ~DistributedPartialResultStep4() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep4Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep4Id id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep4> DistributedPartialResultStep4Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP5LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the fifth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step5Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(LocalCollectionInputId id) const;

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step5LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step5LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step5LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the fourth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP5"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the fifth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep5 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep5)
    /** Default constructor */
    DistributedPartialResultStep5();

    virtual ~DistributedPartialResultStep5() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep5Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep5Id id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep5> DistributedPartialResultStep5Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP6LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the sixth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step6Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(LocalCollectionInputId id) const;

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step6LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step6LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step6LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the sixth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP6"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the sixth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep6 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep6)
    /** Default constructor */
    DistributedPartialResultStep6();

    virtual ~DistributedPartialResultStep6() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep6NumericTableId id) const;

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep6CollectionId id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep6NumericTableId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep6CollectionId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep6> DistributedPartialResultStep6Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP7MASTER"></a>
 * \brief %Input objects for the DBSCAN algorithm in the seventh step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step7Master> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step7MasterCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step7MasterCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step7MasterCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the sixth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP7"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the seventh step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep7 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep7)
    /** Default constructor */
    DistributedPartialResultStep7();

    virtual ~DistributedPartialResultStep7() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep7Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep7Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep7> DistributedPartialResultStep7Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP8LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the eighth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step8Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step8LocalNumericTableInputId id) const;

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step8LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step8LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step8LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step8LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the eighth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP8"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the eighth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep8 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep8)
    /** Default constructor */
    DistributedPartialResultStep8();

    virtual ~DistributedPartialResultStep8() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep8NumericTableId id) const;

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep8CollectionId id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep8NumericTableId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep8CollectionId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep8> DistributedPartialResultStep8Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP9MASTER"></a>
 * \brief %Input objects for the DBSCAN algorithm in the ninth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step9Master> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step9MasterCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step9MasterCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step9MasterCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the ninth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDRESULTSTEP9"></a>
 * \brief Provides methods to access results obtained with the compute() method
 * of the DBSCAN in the ninth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedResultStep9 : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedResultStep9)
    /** Default constructor */
    DistributedResultStep9();

    virtual ~DistributedResultStep9() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedResultStep9Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the new result object
     */
    void set(DistributedResultStep9Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a result of the DBSCAN algorithm
     * \param[in] pres      Partial results of the algorithm
     * \param[in] parameter Algorithm parameter
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * parameter,
                                          const int method);

    /**
     * Checks a result of the DBSCAN algorithm
     * \param[in] pres      Partial results of the algorithm
     * \param[in] parameter Algorithm parameter
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * parameter,
                           int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedResultStep9> DistributedResultStep9Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP9"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the ninth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep9 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep9)
    /** Default constructor */
    DistributedPartialResultStep9();

    virtual ~DistributedPartialResultStep9() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep9Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the new result object
     */
    void set(DistributedPartialResultStep9Id id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep9> DistributedPartialResultStep9Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP10LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the tenth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step10Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step10LocalNumericTableInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step10LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the tenth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP10"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the tenth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep10 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep10)
    /** Default constructor */
    DistributedPartialResultStep10();

    virtual ~DistributedPartialResultStep10() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep10NumericTableId id) const;

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep10CollectionId id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep10NumericTableId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep10CollectionId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep10> DistributedPartialResultStep10Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP11LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the eleventh step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step11Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step11LocalNumericTableInputId id) const;

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step11LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step11LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step11LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step11LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the eleventh step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP11"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the eleventh step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep11 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep11)
    /** Default constructor */
    DistributedPartialResultStep11();

    virtual ~DistributedPartialResultStep11() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep11NumericTableId id) const;

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep11CollectionId id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep11NumericTableId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep11CollectionId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep11> DistributedPartialResultStep11Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP12LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the twelfth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step12Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step12LocalNumericTableInputId id) const;

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step12LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step12LocalNumericTableInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step12LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step12LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the twelfth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP12"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the twelfth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep12 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep12)
    /** Default constructor */
    DistributedPartialResultStep12();

    virtual ~DistributedPartialResultStep12() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedPartialResultStep12Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedPartialResultStep12Id id, const data_management::DataCollectionPtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep12> DistributedPartialResultStep12Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDINPUT_STEP13LOCAL"></a>
 * \brief %Input objects for the DBSCAN algorithm in the thirteenth step
 * of the distributed processing mode
 */
template <>
class DAAL_EXPORT DistributedInput<step13Local> : public daal::algorithms::Input
{
public:
    /** Default constructor */
    DistributedInput();

    /** Copy constructor */
    DistributedInput(const DistributedInput & other) : daal::algorithms::Input(other) {}

    virtual ~DistributedInput() {}

    /**
     * Returns an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step13LocalCollectionInputId id) const;

    /**
     * Sets an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void set(Step13LocalCollectionInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds an input object for the DBSCAN algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the new input object value
     */
    void add(Step13LocalCollectionInputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the parameters and input objects for the DBSCAN algorithm
     * in the thirteenth step of the distributed processing mode
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDRESULTSTEP13"></a>
 * \brief Provides methods to access results obtained with the compute() method
 * of the DBSCAN in the thirteenth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedResultStep13 : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedResultStep13)
    /** Default constructor */
    DistributedResultStep13();

    virtual ~DistributedResultStep13() {}

    /**
     * Returns a result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedResultStep13Id id) const;

    /**
     * Sets a result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the new partial result object
     */
    void set(DistributedResultStep13Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a result of the DBSCAN algorithm
     * \param[in] pres      Partial results of the algorithm
     * \param[in] parameter Algorithm parameter
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * parameter,
                                          const int method);

    /**
     * Checks a result of the DBSCAN algorithm
     * \param[in] pres      Partial results of the algorithm
     * \param[in] parameter Algorithm parameter
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * parameter,
                           int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedResultStep13> DistributedResultStep13Ptr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP13"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 * of the DBSCAN in the thirteenth step of the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResultStep13 : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResultStep13)
    /** Default constructor */
    DistributedPartialResultStep13();

    virtual ~DistributedPartialResultStep13() {}

    /**
     * Returns a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial partial result
     * \return          Value that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(DistributedPartialResultStep13Id id) const;

    /**
     * Sets a partial result of the DBSCAN algorithm
     * \param[in] id    Identifier of the partial partial result
     * \param[in] ptr   Pointer to the new partial partial result object
     */
    void set(DistributedPartialResultStep13Id id, const data_management::NumericTablePtr & ptr);

    /**
     * Allocates memory to store a partial result of the DBSCAN algorithm
     * \param[in] input     Pointer to the input object
     * \param[in] parameter Pointer to the parameter
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Checks a partial result of the DBSCAN algorithm
     * \param[in] input     %Input object for the algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::PartialResult::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedPartialResultStep13> DistributedPartialResultStep13Ptr;

} // namespace interface1

using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

using interface1::DistributedInput;
using interface1::DistributedPartialResultStep1;
using interface1::DistributedPartialResultStep1Ptr;
using interface1::DistributedPartialResultStep2;
using interface1::DistributedPartialResultStep2Ptr;
using interface1::DistributedPartialResultStep3;
using interface1::DistributedPartialResultStep3Ptr;
using interface1::DistributedPartialResultStep4;
using interface1::DistributedPartialResultStep4Ptr;
using interface1::DistributedPartialResultStep5;
using interface1::DistributedPartialResultStep5Ptr;
using interface1::DistributedPartialResultStep6;
using interface1::DistributedPartialResultStep6Ptr;
using interface1::DistributedPartialResultStep7;
using interface1::DistributedPartialResultStep7Ptr;
using interface1::DistributedPartialResultStep8;
using interface1::DistributedPartialResultStep8Ptr;
using interface1::DistributedResultStep9;
using interface1::DistributedResultStep9Ptr;
using interface1::DistributedPartialResultStep9;
using interface1::DistributedPartialResultStep9Ptr;
using interface1::DistributedPartialResultStep10;
using interface1::DistributedPartialResultStep10Ptr;
using interface1::DistributedPartialResultStep11;
using interface1::DistributedPartialResultStep11Ptr;
using interface1::DistributedPartialResultStep12;
using interface1::DistributedPartialResultStep12Ptr;
using interface1::DistributedResultStep13;
using interface1::DistributedResultStep13Ptr;
using interface1::DistributedPartialResultStep13;
using interface1::DistributedPartialResultStep13Ptr;

} // namespace dbscan
/** @} */
} // namespace algorithms
} // namespace daal
#endif
