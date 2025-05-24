/* file: kernel_function_types.h */
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
//  Kernel function parameter structure
//--
*/

#ifndef __KERNEL_FUNCTION_TYPES_H__
#define __KERNEL_FUNCTION_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup kernel_function Kernel Functions
 * \copydoc daal::algorithms::kernel_function
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for computing kernel functions
 */
namespace kernel_function
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION__COMPUTATIONMODE"></a>
 * Mode of computing kernel functions
 */
enum ComputationMode
{
    vectorVector, /*!< Computes the kernel function for given feature vectors Xi and Yj */
    matrixVector, /*!< Computes the kernel function for all the vectors in the set X and a given feature vector Yi */
    matrixMatrix  /*!< Computes the kernel function for all the vectors in the sets X and Y */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION__INPUTID"></a>
 * Available identifiers of input objects of the kernel function algorithm
 */
enum InputId
{
    X, /*!< %Input left data table */
    Y, /*!< %Input right data table */
    lastInputId = Y
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION__RESULTID"></a>
 * Available identifiers of results of the kernel function algorithm
 */
enum ResultId
{
    values, /*!< Table to store results */
    lastResultId = values
};

/**
 * \brief Contains version 1.0 of the oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__KERNEL_FUNCTION__PARAMETERBASE"></a>
 * \brief Optional %input objects for the kernel function algorithm
 *
 * \snippet kernel_function/kernel_function_types.h ParameterBase source code
 */
/* [ParameterBase source code] */
struct DAAL_EXPORT ParameterBase : public daal::algorithms::Parameter
{
    ParameterBase(size_t rowIndexX = 0, size_t rowIndexY = 0, size_t rowIndexResult = 0, ComputationMode computationMode = matrixMatrix);

    size_t rowIndexX;                /*!< Index of the vector in the set X */
    size_t rowIndexY;                /*!< Index of the vector in the set Y */
    size_t rowIndexResult;           /*!< Index of the result of the kernel function computation */
    ComputationMode computationMode; /*!< Mode of computing kernel functions */
};
/* [ParameterBase source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__INPUT"></a>
 * \brief %Input objects for the kernel function algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input();
    Input(const Input & other);
    Input & operator=(const Input & other);
    virtual ~Input() {}

    /**
    * Returns the input object of the kernel function algorithm
    * \param[in] id    Identifier of the input object
    * \return          %Input object that corresponds to the given identifier
    */
    data_management::NumericTablePtr get(InputId id) const;

    /**
    * Sets the input object of the kernel function algorithm
    * \param[in] id    Identifier of the input object
    * \param[in] ptr   Pointer to the input object
    */
    void set(InputId id, const data_management::NumericTablePtr & ptr);

protected:
    services::Status checkCSR() const;

    services::Status checkDense() const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__RESULT"></a>
 * \brief Results obtained with the compute() method of the kernel function algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result)
    Result();

    virtual ~Result() {};
    /**
     * Allocates memory to store results of the kernel function algorithm
     * \param[in] input  Pointer to the structure with the input objects
     * \param[in] par    Pointer to the structure of the algorithm parameters
     * \param[in] method       Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method);

    /**
     * Returns the result of the kernel function algorithm
     * \param[in] id   Identifier of the result
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the kernel function algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the object
     */
    void set(ResultId id, const data_management::NumericTablePtr & ptr);

    /**
    * Checks the result of the kernel function algorithm
    * \param[in] input   %Input objects of the algorithm
    * \param[in] par     %Parameter of the algorithm
    * \param[in] method  Computation method of the algorithm
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
using interface1::ParameterBase;
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;

} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
