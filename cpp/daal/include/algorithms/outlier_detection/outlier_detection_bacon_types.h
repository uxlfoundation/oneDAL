/* file: outlier_detection_bacon_types.h */
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
//  Outlier Detection algorithm parameter structure
//--
*/

#ifndef __OUTLIERDETECTION_BACON_TYPES_H__
#define __OUTLIERDETECTION_BACON_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup bacon_outlier_detection BACON Outlier Detection
 * \copydoc daal::algorithms::bacon_outlier_detection
 * @ingroup analysis
 * @{
 */
/**
* \brief Contains classes for computing the BACON outlier detection
*/
namespace bacon_outlier_detection
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__BACON_OUTLIER_DETECTION__METHOD"></a>
 * Available computation methods for the BACON outlier detection algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Blocked Adaptive Computationally-efficient Outlier Nominators(BACON) method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__BACON_OUTLIER_DETECTION__INITIALIZATIONMETHOD"></a>
 * Available initialization method for the BACON outlier detection algorithm
 */
enum InitializationMethod
{
    baconMedian      = 0, /*!< Median-based method */
    baconMahalanobis = 1  /*!< Mahalanobis distance-based method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__BACON_OUTLIER_DETECTION__INPUTID"></a>
 * Available identifiers of input objects for the BACON outlier detection algorithm
 */
enum InputId
{
    data = 0 /*!< %Input data table */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__BACON_OUTLIER_DETECTION__RESULTID"></a>
 * Available identifiers of the results of the BACON outlier detection algorithm
 */
enum ResultId
{
    weights = 0 /*!< Outlier detection results */
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__BACON_OUTLIER_DETECTION__PARAMETER"></a>
 * \brief Parameters of the outlier detection computation using the baconDense method
 *
 * \snippet outlier_detection/outlier_detection_bacon_types.h ParameterBacon source code
 */
/* [ParameterBacon source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(InitializationMethod initMethod = baconMedian, double alpha = 0.05, double toleranceToConverge = 0.005);

    InitializationMethod initMethod; /*!< Initialization method, \ref InitializationMethod */
    double alpha;                    /*!< One-tailed probability that defines the \f$(1 - \alpha)\f$ quantile
                                                 of the \f$\chi^2\f$ distribution with \f$p\f$ degrees of freedom.
                                                 Recommended value: \f$\alpha / n\f$, where n is the number of observations. */
    double toleranceToConverge;      /*!< Stopping criterion: the algorithm is terminated if the size of the basic subset
                                                 is changed by less than the threshold */
    virtual services::Status check() const DAAL_C11_OVERRIDE;
};
/* [ParameterBacon source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BACON_OUTLIER_DETECTION__INPUT"></a>
 * \brief %Input objects for the BACON outlier detection algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);

    virtual ~Input() {}

    /**
     * Returns input object for the BACON outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input object for the BACON outlier detection algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks input object for the BACON outlier detection algorithm
     * \param[in] par     Algorithm parameters
     * \param[in] method  Computation method for the algorithm
     *
     * \return Status of computations
    */
    services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BACON_OUTLIER_DETECTION__RESULT"></a>
 * \brief Results obtained with the compute() method of the BACON outlier detection algorithm in the %batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory to store the results of the BACON outlier detection algorithm
     * \tparam algorithmFPType  Data type to use for storing results, double or float
     * \param[in] input   Pointer to %Input objects of the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns result of the BACON outlier detection algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the BACON outlier detection algorithm
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
     * Checks the result object of the BACON outlier detection algorithm
     * \param[in] input   Pointer to %Input objects of the algorithm
     * \param[in] par     Pointer to the parameters of the algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
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
using interface1::Parameter;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace bacon_outlier_detection
} // namespace algorithms
} // namespace daal
#endif
