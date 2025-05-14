/* file: distribution_types.h */
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
//  Implementation of distributions.
//--
*/

#ifndef __DISTRIBUTIONS__TYPES__H__
#define __DISTRIBUTIONS__TYPES__H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/engines/engine.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup distributions Distributions
 * \copydoc daal::algorithms::distributions
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for distributions
 */
namespace distributions
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__DISTRIBUTIONS__INPUTID"></a>
 * Available identifiers of input objects for distributions
 */
enum InputId
{
    tableToFill = 0 /*!< Input table to fill with random numbers */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__DISTRIBUTIONS__RESULTID"></a>
 * Available identifiers of results for distributions
 */
enum ResultId
{
    randomNumbers = 0 /*!< Table to store the result */
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__PARAMETERBASE"></a>
 * Parameters of the distributions
 */
class DAAL_EXPORT ParameterBase : public daal::algorithms::Parameter
{
public:
    ParameterBase();
    virtual ~ParameterBase();

    engines::EnginePtr engine; /*!<Pointer to the engine */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__INPUT"></a>
 * \brief %Input objects for distributions
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /**
     * Default constructor
     */
    Input();
    /** Copy constructor */
    Input(const Input & other);

    virtual ~Input();

    /**
     * Returns input table of the distribution
     * \param[in] id    Identifier of the input table
     * \return          %Input table that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input for the distribution
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks an input object for the distribution
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method of the algorithm
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTION__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the distribution
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    /** \brief Constructor */
    Result();

    virtual ~Result();

    /**
     * Allocates memory to store the results of distribution
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Returns result of the distribution
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the distribution
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the result object for the distribution
     * \param[in] input         %Input of the algorithm
     * \param[in] parameter     %Parameter of algorithm
     * \param[in] method        Computation method of the algorithm
     *
     * \return Status of computations
     */
    virtual services::Status check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                   int method) const DAAL_C11_OVERRIDE;

protected:
    using daal::algorithms::interface1::Result::check;

    /** \private */
    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }

    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
using interface1::ParameterBase;
/** @} */
} // namespace distributions
} // namespace algorithms
} // namespace daal
#endif
