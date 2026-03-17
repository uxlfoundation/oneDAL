/* file: precomputed_dense_default_batch_container.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
//  Implementation of precomputed calculation algorithm container.
//--
*/

#ifndef __PRECOMPUTED_DENSE_DEFAULT_BATCH_CONTAINER_H__
#define __PRECOMPUTED_DENSE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/optimization_solver/objective_function/precomputed_batch.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace precomputed
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__PRECOMPUTED__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the objective function with precomputed characteristics.
 *        This class is associated with the Batch class and supports the method of computing
 *        the objective function with precomputed characteristics in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the precomputed objective function, double or float
 * \tparam method           the precomputed objective function computation method
 *
 */
template <typename algorithmFPType, Method method>
class BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the precomputed objective function with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env * daalEnv) {}
    /** Default destructor */
    virtual ~BatchContainer() {}
    /**
     * Runs implementations of the objective function with precomputed characteristics in the batch processing mode
     * \return Status of computations
     */
    services::Status compute() override
    {
        /* empty compute */
        return services::Status();
    }
};
} // namespace internal
} // namespace precomputed
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
