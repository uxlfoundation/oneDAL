/* file: pivoted_qr_types.h */
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
//  Definition of Pivoted QR common types.
//--
*/

#ifndef __PIVOTED_QR_TYPES_H__
#define __PIVOTED_QR_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup pivoted_qr Pivoted QR Decomposition
 * \copydoc daal::algorithms::pivoted_qr
 * @ingroup qr
 * @{
 */
/** \brief Contains classes for computing the pivoted QR decomposition */
namespace pivoted_qr
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__PIVOTED_QR__METHOD"></a>
 * Available methods for computing the results of the pivoted QR algorithm
 */
enum Method
{
    defaultDense = 0 /*!< Default method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PIVOTED_QR__INPUTID"></a>
 * Available types of input objects for the pivoted QR algorithm
 */
enum InputId
{
    data, /*!< Input data table */
    lastInputId = data
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__PIVOTED_QR__RESULTID"></a>
 * Available types of results of the pivoted QR algorithm
 */
enum ResultId
{
    matrixQ,           /*!< Orthogonal Matrix Q */
    matrixR,           /*!< Upper Triangular Matrix R */
    permutationMatrix, /*!< The permutation matrix P overwritten by its details  */
    lastResultId = permutationMatrix
};

/**
 * \brief Contains version 1.0 of oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__PIVOTED_QR__PARAMETER"></a>
 * \brief Parameter for the pivoted QR computation method
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    Parameter(const data_management::NumericTablePtr permutedColumns = data_management::NumericTablePtr());

    data_management::NumericTablePtr permutedColumns; /*!< On entry, if i-th element of permutedColumns != 0,
                                                                  * the i-th column of input matrix is moved  to the beginning of Data * P before
                                                                  * the computation, and fixed in place during the computation.
                                                                  * If i-th element of permutedColumns = 0, the i-th column of input data
                                                                  * is a free column (that is, it may be interchanged during the
                                                                  * computation with any other free column). */
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PIVOTED_QR__INPUT"></a>
 * \brief Input objects for the pivoted QR algorithm in the batch processing mode
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /** Default constructor */
    Input();

    /** Copy constructor */
    Input(const Input & other);

    /** Destructor */
    virtual ~Input() {}

    /**
     * Returns input object for the pivoted QR algorithm
     * \param[in] id    Identifier of the input object
     * \return          Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input object for the pivoted QR algorithm
     * \param[in] id    Identifier of the input object
     * \param[in] value Pointer to the input object
     */
    void set(InputId id, const data_management::NumericTablePtr & value);

    virtual services::Status check(const daal::algorithms::Parameter * par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PIVOTED_QR__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the pivoted QR algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    /** Default constructor */
    Result();
    /** Default destructor */
    virtual ~Result() {}

    /**
     * Allocates memory for storing final results of the pivoted QR algorithm
     * \param[in] input        Pointer to input object
     * \param[in] parameter    Pointer to parameter
     * \param[in] method       Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method);

    /**
     * Returns result of the pivoted QR algorithm
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets data_management::NumericTable to store the result of the pivoted QR algorithm
     * \param[in] id    Identifier of the result
     * \param[in] value Pointer to the storage data_management::NumericTable
     */
    void set(ResultId id, const data_management::NumericTablePtr & value);

    /**
    * Checks the correctness of the result object
    * \param[in] in     Pointer to the input objects structure
    * \param[in] par    Pointer to the structure of the algorithm parameters
    * \param[in] method Computation method
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

} // namespace pivoted_qr
} // namespace algorithms
} // namespace daal
#endif
