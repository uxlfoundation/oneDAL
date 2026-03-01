/* file: df_regression_predict_dense_default_batch_impl.i */
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
//  Common functions for decision forest regression predictions calculation
//--
*/

#ifndef __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __DF_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_predict_dense_default_batch.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/dtrees/regression/dtrees_regression_predict_dense_default_impl.i"
#include "src/services/service_algo_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace prediction
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// PredictRegressionTask
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class PredictRegressionTask : public dtrees::regression::prediction::internal::PredictRegressionTaskBase<algorithmFPType, cpu>
{
public:
    typedef dtrees::regression::prediction::internal::PredictRegressionTaskBase<algorithmFPType, cpu> super;
    PredictRegressionTask(const NumericTable * x, NumericTable * y) : super(x, y) {}

    services::Status run(const decision_forest::regression::internal::ModelImpl * m, services::HostAppIface * pHostApp);
};

//////////////////////////////////////////////////////////////////////////////////////////
// PredictKernel
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictKernel<algorithmFPType, method, cpu>::compute(services::HostAppIface * pHostApp, const NumericTable * x,
                                                                      const regression::Model * m, NumericTable * r)
{
    DAAL_PROFILER_TASK(Regression : PredictKernel::compute);
    const daal::algorithms::decision_forest::regression::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::decision_forest::regression::internal::ModelImpl *>(m);
    PredictRegressionTask<algorithmFPType, cpu> task(x, r);
    return task.run(pModel, pHostApp);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictRegressionTask<algorithmFPType, cpu>::run(const decision_forest::regression::internal::ModelImpl * m,
                                                                  services::HostAppIface * pHostApp)
{
    DAAL_PROFILER_TASK(Regression : PredictRegressionTask::run);
    DAAL_CHECK_MALLOC(this->_featHelper.init(*this->_data));
    const auto nTreesTotal = m->size();
    this->_aTree.reset(nTreesTotal);
    DAAL_CHECK_MALLOC(this->_aTree.get());
    for (size_t i = 0; i < nTreesTotal; ++i) this->_aTree[i] = m->at(i);
    const algorithmFPType div = algorithmFPType(1) / algorithmFPType(nTreesTotal);
    return super::run(pHostApp, div);
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
