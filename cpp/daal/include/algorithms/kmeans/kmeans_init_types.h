/* file: kmeans_init_types.h */
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
//  Implementation of the interface for initializing K-Means algorithm interface.
//--
*/

#ifndef __KMEANS_INIT_TYPES_H__
#define __KMEANS_INIT_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/data_collection.h"
#include "services/daal_defines.h"
#include "algorithms/engines/mt19937/mt19937.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
/**
 * @defgroup kmeans K-means Clustering
 * \copydoc daal::algorithms::kmeans
 * @ingroup analysis
 * @defgroup kmeans_init Initialization
 * \copydoc daal::algorithms::kmeans::init
 * @ingroup kmeans
 * @{
 */
/** \brief Contains classes for computing initial centroids for K-Means algorithm */
namespace init
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__METHOD"></a>
 * Available methods for computing initial centroids for K-Means algorithm
 */
enum Method
{
    deterministicDense = 0, /*!< Default: uses first nClusters points as initial centroids */
    defaultDense       = 0, /*!< Synonym of deterministicDense */
    randomDense        = 1, /*!< Uses random nClusters points as initial centroids */
    plusPlusDense      = 2, /*!< Kmeans++ algorithm by Arthur and Vassilvitskii (2007):
                                 http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf [1]
                                 the first center is selected at random, each subsequent center is
                                 selected with a probability proportional to its contribution to the overall error */
    parallelPlusDense  = 3, /*!< Kmeans|| algorithm: scalable Kmeans++ by Bahmani et al. (2012)
                                 http://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf [2]*/
    deterministicCSR   = 4, /*!< Uses first nClusters points as initial centroids for data in a CSR numeric table */
    randomCSR          = 5, /*!< Uses random nClusters points as initial centroids for data in a CSR numeric table */
    plusPlusCSR        = 6, /*!< Kmeans++ algorithm Arthur and Vassilvitskii (2007)
                                 http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf [1]
                                 for the data in a CSR numeric table:
                                 the first center is selected at random, each subsequent center is
                                 selected with a probability proportional to its contribution to the overall error */
    parallelPlusCSR    = 7  /*!< Kmeans|| algorithm: scalable Kmeans++ by Bahmani et al. (2012)
                                 http://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf [2]
                                 for the data in a CSR numeric table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__INPUTID"></a>
 * \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm
 */
enum InputId
{
    data, /*!< %Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2MASTERINPUTID"></a>
 * \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm in the distributed processing mode
 */
enum DistributedStep2MasterInputId
{
    partialResults, /*!< Collection of partial results computed on local nodes */
    lastDistributedStep2MasterInputId = partialResults
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDLOCALPLUSPLUSINPUTDATAID"></a>
* \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm
*        used with plusPlus and parallelPlus methods only on a local node.
*/
enum DistributedLocalPlusPlusInputDataId
{
    internalInput =
        lastDistributedStep2MasterInputId + 1, /*!< %DataCollection with internal algorithm data calculated by previous steps on this node*/
    lastDistributedLocalPlusPlusInputDataId = internalInput
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2LOCALPLUSPLUSINPUTID"></a>
* \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm
*        used with plusPlus and parallelPlus methods only on the 2nd step on a local node.
*/
enum DistributedStep2LocalPlusPlusInputId
{
    inputOfStep2 = lastDistributedLocalPlusPlusInputDataId
                   + 1, /*!< %Numeric table with the new centroids calculated by previous steps of initialization algorithm */
    lastDistributedStep2LocalPlusPlusInputId = inputOfStep2
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP3MASTERPLUSPLUSINPUTID"></a>
* \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm
*        used with plusPlus and parallelPlus methods only on the 3rd step on a master node.
*/
enum DistributedStep3MasterPlusPlusInputId
{
    inputOfStep3FromStep2, /*!< %Numeric table with the data calculated on step2 on local nodes*/
    lastDistributedStep3MasterPlusPlusInputId = inputOfStep3FromStep2
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP4LOCALPLUSPLUSINPUTID"></a>
* \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm
*        used with plusPlus and parallelPlus methods only on a local node.
*/
enum DistributedStep4LocalPlusPlusInputId
{
    inputOfStep4FromStep3 = lastDistributedLocalPlusPlusInputDataId + 1, /*!< %Numeric table with the data calculated on step3 on master node */
    lastDistributedStep4LocalPlusPlusInputId = inputOfStep4FromStep3
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP5MASTERPLUSPLUSINPUTID"></a>
* \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm
*        used with parallelPlus method only on a master node.
*/
enum DistributedStep5MasterPlusPlusInputId
{
    inputCentroids,        /*!< %DataCollection of NumericTables with the new centroids */
    inputOfStep5FromStep2, /*!< %DataCollection of NumericTables with the new centroids rating */
    lastDistributedStep5MasterPlusPlusInputId = inputOfStep5FromStep2
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP5MASTERPLUSPLUSINPUTDATAID"></a>
* \brief Available identifiers of input objects for computing initial centroids for K-Means algorithm
*        used with parallelPlus methods only on the 5th step on a master node.
*/
enum DistributedStep5MasterPlusPlusInputDataId
{
    inputOfStep5FromStep3 = lastDistributedStep5MasterPlusPlusInputId + 1, /*!< %Service data generated as the output of step3Master */
    lastDistributedStep5MasterPlusPlusInputDataId = inputOfStep5FromStep3
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__PARTIALRESULTID"></a>
* \brief Available identifiers of partial results of computing initial centroids for K-Means algorithm in the distributed processing mode
*/
enum PartialResultId
{
    partialCentroids,                   /*!< Table with the sum of observations assigned to centroids */
    partialClusters = partialCentroids, /*!< Table with the sum of observations assigned to centroids \DAAL_DEPRECATED */
    partialClustersNumber,              /*!< Table with the number of observations assigned to centroids \DAAL_DEPRECATED */
    lastPartialResultId = partialClustersNumber
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2LOCALPLUSPLUSPARTIALRESULTID"></a>
* \brief Available identifiers of partial results of computing initial centroids for K-Means algorithm in the distributed processing mode
*        used with plusPlus and parallelPlus methods only on the 2nd step on a local node.
*/
enum DistributedStep2LocalPlusPlusPartialResultId
{
    outputOfStep2ForStep3, /*!< %Numeric table containing output from step 2 on the local node used by step 3 on a master node*/
    outputOfStep2ForStep5, /*!< %Numeric table containing output from step 2 on the local node used by step 5 on a master node*/
    lastDistributedStep2LocalPlusPlusPartialResultId = outputOfStep2ForStep5
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2LOCALPLUSPLUSPARTIALRESULTDATAID"></a>
* \brief Available identifiers of partial results of computing initial centroids for K-Means algorithm in the distributed processing mode
*        used with plusPlus and parallelPlus methods only on the 2nd step on a local node.
*/
enum DistributedStep2LocalPlusPlusPartialResultDataId
{
    internalResult = lastDistributedStep2LocalPlusPlusPartialResultId
                     + 1, /*!< %DataCollection with internal algorithm data required as an input for the future steps on the node*/
    lastDistributedStep2LocalPlusPlusPartialResultDataId = internalResult
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP3MASTERPLUSPLUSPARTIALRESULTID"></a>
* \brief Available identifiers of partial results of computing initial centroids for K-Means algorithm in the distributed processing mode
*        used with plusPlus and parallelPlus methods only on the 3rd step on a master node.
*/
enum DistributedStep3MasterPlusPlusPartialResultId
{
    outputOfStep3ForStep4, /*!< %KeyValueDataCollection with the input for local nodes on step 4 */
    lastDistributedStep3MasterPlusPlusPartialResultId = outputOfStep3ForStep4
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP3MASTERPLUSPLUSPARTIALRESULTDATAID"></a>
* \brief Available identifiers of partial results of computing initial centroids for K-Means algorithm in the distributed processing mode
*        used with parallelPlus method only on the 3rd step on a master node.
*/
enum DistributedStep3MasterPlusPlusPartialResultDataId
{
    rngState =
        lastDistributedStep3MasterPlusPlusPartialResultId + 1, /*!< %Service data generated as the output of step3Master to be used in step5Master*/
    outputOfStep3ForStep5 = rngState,                          /*!< %Service data generated as the output of step3Master to be used in step5Master*/
    lastDistributedStep3MasterPlusPlusPartialResultDataId = outputOfStep3ForStep5
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP4LOCALPLUSPLUSPARTIALRESULTID"></a>
* \brief Available identifiers of partial results of computing initial centroids for K-Means algorithm in the distributed processing mode
*        used with plusPlus and parallelPlus methods only on the 4th step on a local node.
*/
enum DistributedStep4LocalPlusPlusPartialResultId
{
    outputOfStep4, /*!< %NumericTable with the new centroids calculated on step 4 on the local node */
    lastDistributedStep4LocalPlusPlusPartialResultId = outputOfStep4
};

/**
* <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP5MASTERPLUSPLUSPARTIALRESULTID"></a>
* \brief Available identifiers of partial results of computing initial centroids for K-Means algorithm in the distributed processing mode
*        used with parallelPlus method only on the 5th step on a master node.
*/
enum DistributedStep5MasterPlusPlusPartialResultId
{
    candidates, /*!< %NumericTable with the new centroids calculated on the previous steps */
    weights,    /*!< %NumericTable with the weights of the new centroids calculated on the previous steps */
    lastDistributedStep5MasterPlusPlusPartialResultId = weights
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KMEANS__INIT__RESULTID"></a>
 * \brief Available identifiers of the results of computing initial centroids for K-Means algorithm
 */
enum ResultId
{
    centroids, /*!< Table for cluster centroids */
    lastResultId = centroids
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__KMEANS__INIT__PARAMETER"></a>
 * \brief Base classes parameters for computing initial centroids for K-Means algorithm
 *
 * \snippet kmeans/kmeans_init_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /**
     *  Parameter constructor
     *  \param[in] _nClusters     Number of clusters
     *  \param[in] _offset        Offset in the total data set specifying the start of a block stored on a given local node
     *  \param[in] _seed          Seed for generating random numbers for the initialization \DAAL_DEPRECATED_USE{ engine }
     */
    Parameter(size_t _nClusters, size_t _offset = 0, size_t _seed = 777777);

    /**
     * Constructs parameters of the algorithm that computes initial centroids for K-Means algorithm
     * by copying another parameters object
     * \param[in] other    Parameters of K-Means algorithm
     */
    Parameter(const Parameter & other);

    size_t nClusters;  /*!< Number of clusters */
    size_t nRowsTotal; /*!< Total number of rows in the data set  */
    size_t offset;     /*!< Offset in the total data set specifying the start of a block stored on a given local node */
    size_t seed;       /*!< Seed for generating random numbers for the initialization \DAAL_DEPRECATED_USE{ engine } */

    double oversamplingFactor; /*!< Kmeans|| only. A fraction of nClusters being chosen in each of nRounds of kmeans||.\
                                                   L = nClusters* oversamplingFactor points are sampled in a round. */
    size_t nRounds;            /*!< Kmeans|| only. Number of rounds for k-means||. (oversamplingFactor*nRounds) > 1 is a requirement.*/

    engines::EnginePtr engine; /*!< Engine to be used for generating random numbers for the initialization */

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INPUTIFACE"></a>
 * \brief Interface for K-Means initialization batch and distributed Input classes
 */
class DAAL_EXPORT InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {};

    virtual size_t getNumberOfFeatures() const = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__INPUT"></a>
 * \brief %Input objects for computing initial centroids for K-Means algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();
    virtual ~Input() {}

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Returns the number of features in the Input data table
    * \return Number of features in the Input data table
    */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
    * Checks an input object for computing initial centroids for K-Means algorithm
    * \param[in] par     %Input object
    * \param[in] method  Method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    Input(size_t nElements);
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__PARTIALRESULT"></a>
 * \brief Partial results obtained with the compute() method of K-Means algorithm in the batch processing mode
 */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult)
    PartialResult();

    virtual ~PartialResult() {};

    /**
     * Allocates memory to store partial results of computing initial centroids for K-Means algorithm
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns a partial result of computing initial centroids for K-Means algorithm
     * \param[in] id   Identifier of the partial result
     * \return         Partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialResultId id) const;

    /**
     * Sets a partial result of computing initial centroids for K-Means algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the object
     */
    void set(PartialResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Returns the number of features in the result table of K-Means algorithm
     * \return Number of features in the result table of K-Means algorithm
     */
    size_t getNumberOfFeatures() const;

    /**
     * Checks a partial result of computing initial centroids for K-Means algorithm
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks a partial result of computing initial centroids for K-Means algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<PartialResult> PartialResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__RESULT"></a>
 * \brief Results obtained with the compute() method that computes initial centroids
 *  for K-Means algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of computing initial centroids for K-Means algorithm
     * \param[in] input        Pointer to the input structure
     * \param[in] parameter    Pointer to the parameter structure
     * \param[in] method       Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Allocates memory to store the results of computing initial centroids for K-Means algorithm
     * \param[in] partialResult Pointer to the partial result structure
     * \param[in] parameter     Pointer to the parameter structure
     * \param[in] method        Computation method of the algorithm
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult * partialResult, const daal::algorithms::Parameter * parameter,
                                          const int method);

    /**
     * Returns the result of computing initial centroids for K-Means algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of computing initial centroids for K-Means algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the result of computing initial centroids for K-Means algorithm
     * \param[in] input   %Input object for the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks the result of computing initial centroids for K-Means algorithm
     * \param[in] pres    Partial results of the algorithm
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::PartialResult * pres, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2MASTERINPUT"></a>
 * \brief %Input objects for computing initials clusters for K-Means
 *  algorithm in the second step of the distributed processing mode
 */
class DAAL_EXPORT DistributedStep2MasterInput : public InputIface
{
public:
    DistributedStep2MasterInput();

    virtual ~DistributedStep2MasterInput() {}

    /**
     * Returns an input object for computing initial centroids for K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(DistributedStep2MasterInputId id) const;

    /**
     * Sets an input object for computing initial centroids for K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(DistributedStep2MasterInputId id, const data_management::DataCollectionPtr & ptr);

    /**
     * Adds a value to the data collection of input objects for computing initial centroids for K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] id    Identifier of the parameter
     * \param[in] value Pointer to the new parameter value
     */
    void add(DistributedStep2MasterInputId id, const PartialResultPtr & value);

    /**
     * Returns the number of features in the Input data table in the second step of the distributed processing mode
     * \return Number of features in the Input data table
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
     * Checks an input object for computing initial centroids for K-Means algorithm
     * in the second step of the distributed processing mode
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-STRUCT-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2LOCALPLUSPLUSPARAMETER"></a>
* \brief Parameters for computing initial centroids for K-Means algorithm
*/
struct DAAL_EXPORT DistributedStep2LocalPlusPlusParameter : public Parameter
{
    /**
    *  Main constructor
    */
    DistributedStep2LocalPlusPlusParameter(size_t _nClusters, bool bFirstIteration);

    /**
    * Constructs parameters of the algorithm that computes initial centroids for K-Means algorithm
    * by copying another parameters object
    * \param[in] other    Parameters of K-Means algorithm
    */
    DistributedStep2LocalPlusPlusParameter(const DistributedStep2LocalPlusPlusParameter & other);

    bool firstIteration;         /*!< True if step2 is called for the first time */
    bool outputForStep5Required; /*!< True if the last iteration of parallelPlus algorithm processing is performed */
    services::Status check() const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2LOCALPLUSPLUSINPUT"></a>
* \brief Interface for K-Means initialization distributed Input classes
*        used with plusPlus and parallelPlus methods only on the 2nd step on a local node.
*/
class DAAL_EXPORT DistributedStep2LocalPlusPlusInput : public Input
{
public:
    DistributedStep2LocalPlusPlusInput();
    DistributedStep2LocalPlusPlusInput(const DistributedStep2LocalPlusPlusInput & o);

    virtual ~DistributedStep2LocalPlusPlusInput() {}

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(DistributedLocalPlusPlusInputDataId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(DistributedLocalPlusPlusInputDataId id, const data_management::DataCollectionPtr & ptr);

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DistributedStep2LocalPlusPlusInputId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(DistributedStep2LocalPlusPlusInputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks an input object for computing initial centroids for K-Means algorithm
    * \param[in] par     %Input object
    * \param[in] method  Method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP3MASTERPLUSPLUSINPUT"></a>
* \brief Interface for K-Means distributed Input classes
*        used with plusPlus and parallelPlus methods only on the 3rd step on a master node.
*/
class DAAL_EXPORT DistributedStep3MasterPlusPlusInput : public daal::algorithms::Input
{
public:
    DistributedStep3MasterPlusPlusInput();
    DistributedStep3MasterPlusPlusInput(const DistributedStep3MasterPlusPlusInput & o);

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::KeyValueDataCollectionPtr get(DistributedStep3MasterPlusPlusInputId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(DistributedStep3MasterPlusPlusInputId id, const data_management::KeyValueDataCollectionPtr & ptr);

    /**
    * Add an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] key   Identifier of the node this object comes from
    * \param[in] ptr   Pointer to the input object
    */
    void add(DistributedStep3MasterPlusPlusInputId id, size_t key, const data_management::NumericTablePtr & ptr);

    /**
    * Checks an input object for computing initial centroids for K-Means algorithm
    * \param[in] par     %Input object
    * \param[in] method  Method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP4LOCALPLUSPLUSINPUT"></a>
* \brief Interface for K-Means distributed Input classes
*        used with plusPlus and parallelPlus methods only on the 4th step on a local node.
*/
class DAAL_EXPORT DistributedStep4LocalPlusPlusInput : public Input
{
public:
    DistributedStep4LocalPlusPlusInput();
    DistributedStep4LocalPlusPlusInput(const DistributedStep4LocalPlusPlusInput & o);

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(DistributedLocalPlusPlusInputDataId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(DistributedLocalPlusPlusInputDataId id, const data_management::DataCollectionPtr & ptr);

    /**
    * Returns input objects for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DistributedStep4LocalPlusPlusInputId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(DistributedStep4LocalPlusPlusInputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks an input object for computing initial centroids for K-Means algorithm
    * \param[in] par     %Input object
    * \param[in] method  Method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP5MASTERPLUSPLUSINPUT"></a>
* \brief Interface for K-Means distributed Input classes
*/
class DAAL_EXPORT DistributedStep5MasterPlusPlusInput : public daal::algorithms::Input
{
public:
    DistributedStep5MasterPlusPlusInput();
    DistributedStep5MasterPlusPlusInput(const DistributedStep5MasterPlusPlusInput & o);

    virtual ~DistributedStep5MasterPlusPlusInput() {}

    /**
    * Returns an input object for computing initial centroids for K-Means algorithm
    * in the 5th step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(DistributedStep5MasterPlusPlusInputId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * in the 5th step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep5MasterPlusPlusInputId id, const data_management::DataCollectionPtr & ptr);

    /**
    * Adds a value to the data collection of input objects for computing initial centroids for K-Means algorithm
    * in the 5th step of the distributed processing mode
    * \param[in] id    Identifier of the parameter
    * \param[in] value Pointer to the new parameter value
    */
    void add(DistributedStep5MasterPlusPlusInputId id, const data_management::NumericTablePtr & value);

    /**
    * Returns an input object for computing initial centroids for K-Means algorithm
    * in the 5th step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::SerializationIfacePtr get(DistributedStep5MasterPlusPlusInputDataId id) const;

    /**
    * Sets an input object for computing initial centroids for K-Means algorithm
    * in the 5th step of the distributed processing mode
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep5MasterPlusPlusInputDataId id, const data_management::SerializationIfacePtr & ptr);

    /**
    * Checks an input object for computing initial centroids for K-Means algorithm
    * in the 5th step of the distributed processing mode
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP2LOCALPLUSPLUSPARTIALRESULT"></a>
* \brief Partial results obtained with the compute() method of K-Means algorithm in the distributed processing mode
*/
class DAAL_EXPORT DistributedStep2LocalPlusPlusPartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedStep2LocalPlusPlusPartialResult)
    DistributedStep2LocalPlusPlusPartialResult();

    virtual ~DistributedStep2LocalPlusPlusPartialResult() {};

    /**
    * Allocates memory to store partial results of computing initial centroids for K-Means algorithm
    * \param[in] input     Pointer to the input structure
    * \param[in] parameter Pointer to the parameter structure
    * \param[in] method    Computation method of the algorithm
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
    * Returns a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id   Identifier of the partial result
    * \return         Partial result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DistributedStep2LocalPlusPlusPartialResultId id) const;

    /**
    * Sets a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the partial result
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep2LocalPlusPlusPartialResultId id, const data_management::NumericTablePtr & ptr);

    /**
    * Returns a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id   Identifier of the partial result
    * \return         Partial result that corresponds to the given identifier
    */
    data_management::DataCollectionPtr get(DistributedStep2LocalPlusPlusPartialResultDataId id) const;

    /**
    * Sets a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the partial result
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep2LocalPlusPlusPartialResultDataId id, const data_management::DataCollectionPtr & ptr);

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Initializes the partial result data
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedStep2LocalPlusPlusPartialResult> DistributedStep2LocalPlusPlusPartialResultPtr;

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP3MASTERPLUSPLUSPARTIALRESULT"></a>
* \brief Partial results obtained with the compute() method of K-Means algorithm in the distributed processing mode
*/
class DAAL_EXPORT DistributedStep3MasterPlusPlusPartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedStep3MasterPlusPlusPartialResult)
    DistributedStep3MasterPlusPlusPartialResult();

    virtual ~DistributedStep3MasterPlusPlusPartialResult() {};

    /**
    * Allocates memory to store partial results of computing initial centroids for K-Means algorithm
    * \param[in] input     Pointer to the input structure
    * \param[in] parameter Pointer to the parameter structure
    * \param[in] method    Computation method of the algorithm
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
    * Returns a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id   Identifier of the partial result
    * \return         Partial result that corresponds to the given identifier
    */
    data_management::KeyValueDataCollectionPtr get(DistributedStep3MasterPlusPlusPartialResultId id) const;

    /**
    * Returns a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id   Identifier of the partial result
    * \param[in] key  Identifier of the node this partial result comes from
    * \return         Partial result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DistributedStep3MasterPlusPlusPartialResultId id, size_t key) const;

    /**
    * Returns a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id   Identifier of the partial result
    * \return         Partial result that corresponds to the given identifier
    */
    data_management::SerializationIfacePtr get(DistributedStep3MasterPlusPlusPartialResultDataId id) const;

    /**
    * Sets a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the partial result
    * \param[in] key   Identifier of the node this partial result comes from
    * \param[in] ptr   Pointer to the object
    */
    void add(DistributedStep3MasterPlusPlusPartialResultId id, size_t key, const data_management::NumericTablePtr & ptr);

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Initializes the partial result data
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    void initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method);

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedStep3MasterPlusPlusPartialResult> DistributedStep3MasterPlusPlusPartialResultPtr;

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP4LOCALPLUSPLUSPARTIALRESULT"></a>
* \brief Partial results obtained with the compute() method of K-Means algorithm in the distributed processing mode
*/
class DAAL_EXPORT DistributedStep4LocalPlusPlusPartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedStep4LocalPlusPlusPartialResult)
    DistributedStep4LocalPlusPlusPartialResult();

    virtual ~DistributedStep4LocalPlusPlusPartialResult() {};

    /**
    * Allocates memory to store partial results of computing initial centroids for K-Means algorithm
    * \param[in] input     Pointer to the input structure
    * \param[in] parameter Pointer to the parameter structure
    * \param[in] method    Computation method of the algorithm
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
    * Returns a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id   Identifier of the partial result
    * \return         Partial result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DistributedStep4LocalPlusPlusPartialResultId id) const;

    /**
    * Sets a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the partial result
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep4LocalPlusPlusPartialResultId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedStep4LocalPlusPlusPartialResult> DistributedStep4LocalPlusPlusPartialResultPtr;

/**
* <a name="DAAL-CLASS-ALGORITHMS__KMEANS__INIT__DISTRIBUTEDSTEP5MASTERPLUSPLUSPARTIALRESULT"></a>
* \brief Partial results obtained with the compute() method of K-Means algorithm in the distributed processing mode
*/
class DAAL_EXPORT DistributedStep5MasterPlusPlusPartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedStep5MasterPlusPlusPartialResult)
    DistributedStep5MasterPlusPlusPartialResult();

    virtual ~DistributedStep5MasterPlusPlusPartialResult() {};

    /**
    * Allocates memory to store partial results of computing initial centroids for K-Means algorithm
    * \param[in] input     Pointer to the input structure
    * \param[in] parameter Pointer to the parameter structure
    * \param[in] method    Computation method of the algorithm
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
    * Returns a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id   Identifier of the partial result
    * \return         Partial result that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(DistributedStep5MasterPlusPlusPartialResultId id) const;

    /**
    * Sets a partial result of computing initial centroids for K-Means algorithm
    * \param[in] id    Identifier of the partial result
    * \param[in] ptr   Pointer to the object
    */
    void set(DistributedStep5MasterPlusPlusPartialResultId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] input   %Input object for the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks a partial result of computing initial centroids for K-Means algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<DistributedStep5MasterPlusPlusPartialResult> DistributedStep5MasterPlusPlusPartialResultPtr;

} // namespace interface1

namespace interface2
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__KMEANS__INIT__BATCH__PARAMETER"></a>
 * \brief Parameters for computing initial centroids for K-Means algorithm of the batch mode
 */
struct DAAL_EXPORT Parameter : public interface1::Parameter
{
    Parameter(size_t _nClusters, size_t _offset = 0, size_t _seed = 777777);

    /**
     * Constructs parameters of the algorithm that computes initial centroids for K-Means algorithm
     * by copying another parameters object
     * \param[in] other    Parameters of K-Means algorithm
     */
    Parameter(const Parameter & other);

    size_t nTrials; /*!< Kmeans++ only. The number of trials to generate all clusters but the first initial cluster. */

    services::Status check() const DAAL_C11_OVERRIDE;
};

} // namespace interface2

using interface2::Parameter;
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::Result;
using interface1::ResultPtr;
using interface1::DistributedStep2MasterInput;
using interface1::DistributedStep2LocalPlusPlusParameter;
using interface1::DistributedStep2LocalPlusPlusInput;
using interface1::DistributedStep3MasterPlusPlusInput;
using interface1::DistributedStep4LocalPlusPlusInput;
using interface1::DistributedStep5MasterPlusPlusInput;
using interface1::DistributedStep2LocalPlusPlusPartialResult;
using interface1::DistributedStep2LocalPlusPlusPartialResultPtr;
using interface1::DistributedStep3MasterPlusPlusPartialResult;
using interface1::DistributedStep3MasterPlusPlusPartialResultPtr;
using interface1::DistributedStep4LocalPlusPlusPartialResult;
using interface1::DistributedStep4LocalPlusPlusPartialResultPtr;
using interface1::DistributedStep5MasterPlusPlusPartialResult;
using interface1::DistributedStep5MasterPlusPlusPartialResultPtr;

} // namespace init
/** @} */
} // namespace kmeans
} // namespace algorithms
} // namespace daal
#endif
