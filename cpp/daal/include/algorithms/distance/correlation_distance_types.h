/* file: correlation_distance_types.h */
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
//  Implementation of correlation distance algorithm interface.
//--
*/

#ifndef __CORDISTANCE_TYPES_H__
#define __CORDISTANCE_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/symmetric_matrix.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup correlation_distance Correlation Distance Matrix
 * \copydoc daal::algorithms::correlation_distance
 * @ingroup analysis
 * @{
 */
/**
* \brief Contains classes for computing the correlation distance
*/
namespace correlation_distance
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__CORRELATION_DISTANCE__METHOD"></a>
 * Available methods for computing the correlation distance
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__CORRELATION_DISTANCE__INPUTID"></a>
 * Available identifiers of input objects for the correlation distance algorithm
 */
enum InputId
{
    data, /*!< %Input data table */
    lastInputId = data
};
/**
 * <a name="DAAL-ENUM-ALGORITHMS__CORRELATION_DISTANCE__RESULTID"></a>
 * Available identifiers of results for the correlation distance algorithm
 */
enum ResultId
{
    correlationDistance, /*!< Table to store the result.*/
    lastResultId = correlationDistance
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__CORRELATION_DISTANCE__INPUT"></a>
 * \brief %Input objects for the correlation distance algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other) : daal::algorithms::Input(other) {}

    virtual ~Input() {}

    /**
    * Returns the input object of the correlation distance algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets the input object of the correlation distance algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the object
    */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks the parameters of the correlation distance algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CORRELATION_DISTANCE__RESULT"></a>
 * \brief Results obtained with compute() method of the correlation distance algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the correlation distance algorithm
     * \param[in] input  Pointer to input structure
     * \param[in] par    Pointer to parameter structure
     * \param[in] method Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Returns the result of the correlation distance algorithm
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the correlation distance algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the value
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks the result of the correlation distance algorithm
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of the algorithm
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
/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace correlation_distance
} // namespace algorithms
} // namespace daal
#endif
