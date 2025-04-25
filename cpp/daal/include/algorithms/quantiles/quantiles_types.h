/* file: quantiles_types.h */
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
//  Definition of common types of quantiles.
//--
*/

#ifndef __QUANTILES_TYPES_H__
#define __QUANTILES_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
* @defgroup quantiles Quantile
* \copydoc daal::algorithms::quantiles
* @ingroup analysis
* @{
*/
/**
 * \brief Contains classes to run the quantile algorithms
 */
namespace quantiles
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__QUANTILES__METHOD"></a>
 * Available methods for quantiles computation
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method. Works with all types of input numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QUANTILES__INPUTID"></a>
 * Available identifiers of input objects for the quantiles algorithm
 */
enum InputId
{
    data, /*!< %Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__QUANTILES__RESULTID"></a>
 * Available identifiers of results of the quantiles algorithm
 */
enum ResultId
{
    quantiles, /*!< Values of quantiles */
    lastResultId = quantiles
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__QUANTILES__PARAMETER"></a>
 * \brief Parameters of the quantiles algorithm
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(const data_management::NumericTablePtr quantileOrders = data_management::NumericTablePtr());
    data_management::NumericTablePtr quantileOrders; /*!< Numeric table with quantile orders. Default value is 0.5 (median) */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__INPUT"></a>
 * \brief %Input objects for the quantiles algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);

    virtual ~Input() {}

    /**
     * Returns an input object for the quantiles algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets the input object of the quantiles algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Check the correctness of the %Input object
     * \param[in] parameter Pointer to the parameters structure
     * \param[in] method    Algorithm computation method
     */
    virtual services::Status check(const daal::algorithms::Parameter * parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__QUANTILES__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the
 *        quantiles algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store final results of the quantile algorithms
     * \param[in] input     Input objects for the quantiles algorithm
     * \param[in] parameter Parameters of the quantiles algorithm
     * \param[in] method    Algorithm computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns the final result of the quantiles algorithm
     * \param[in] id   Identifier of the final result, \ref ResultId
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the Result object of the quantiles algorithm
     * \param[in] id        Identifier of the Result object
     * \param[in] value     Pointer to the Result object
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
     * Checks the correctness of the Result object
     * \param[in] in     Pointer to the object
     * \param[in] par    Pointer to the parameters structure
     * \param[in] method Algorithm computation method
     */
    virtual services::Status check(const daal::algorithms::Input * in, const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;

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
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace quantiles
} // namespace algorithms
} // namespace daal
#endif
