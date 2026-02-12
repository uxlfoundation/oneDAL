/* file: sum_of_functions_batch.h */
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
//  Implementation of the Sum of functions types.
//--
*/

#ifndef __SUM_OF_FUNCTIONS_BATCH_H__
#define __SUM_OF_FUNCTIONS_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/objective_function/objective_function_batch.h"
#include "algorithms/optimization_solver/objective_function/sum_of_functions_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace sum_of_functions
{
namespace interface2
{
/**
 * @defgroup sum_of_functions_batch Batch
 * @ingroup sum_of_functions
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SUM_OF_FUNCTIONS__BATCH"></a>
 * \brief Interface for computing the Sum of functions in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SUM_OF_FUNCTIONS-ALGORITHM">the Sum of functions description and usage models</a> -->
 *
 * \par Enumerations
 *      - \ref InputId  Identifiers of input objects for the Sum of functions
 *      - \ref objective_function::ResultId Result identifiers for the Sum of functions
 *
 * \par References
 *      - \ref interface1::Input class
 *      - \ref interface1::Result class
 */
class DAAL_EXPORT Batch : public objective_function::Batch
{
public:
    typedef objective_function::Batch super;

    typedef algorithms::optimization_solver::sum_of_functions::Input InputType;
    typedef algorithms::optimization_solver::sum_of_functions::Parameter ParameterType;
    typedef super::ResultType ResultType;

    /**
     *  Main constructor
     */
    Batch(size_t numberOfTerms, InputType * sumOfFunctionsInput, ParameterType * sumOfFunctionsParameter)
        : sumOfFunctionsParameter(sumOfFunctionsParameter), sumOfFunctionsInput(sumOfFunctionsInput)
    {
        initialize();
        if (sumOfFunctionsParameter != NULL)
        {
            sumOfFunctionsParameter->numberOfTerms = numberOfTerms;
        }
    }

    /**
     * Constructs the Sum of functions by copying input objects and parameters
     * of another the Sum of functions
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch & other)
        : objective_function::Batch(), sumOfFunctionsParameter(other.sumOfFunctionsParameter), sumOfFunctionsInput(other.sumOfFunctionsInput)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Returns a pointer to the newly allocated Sum of functions with a copy of input objects
     * of this Sum of functions
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch> clone() const { return services::SharedPtr<Batch>(cloneImpl()); }

    ParameterType * sumOfFunctionsParameter; /*!< Pointer to the parameter to use one object in inherited class */
    InputType * sumOfFunctionsInput;         /*!< Pointer to the input to use one object in inherited class */

protected:
    virtual Batch * cloneImpl() const DAAL_C11_OVERRIDE = 0;

    void initialize() {}

private:
    Batch & operator=(const Batch &);
};
typedef services::SharedPtr<Batch> BatchPtr;

/** @} */
} // namespace interface2
using interface2::Batch;
using interface2::BatchPtr;

} // namespace sum_of_functions
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
