/* file: df_classification_training_types_result.h */
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
//  Implementation of decision forest algorithm classes.
//--
*/

#include "algorithms/decision_forest/decision_forest_classification_training_types.h"
#include "src/algorithms/engines/mt2203/mt2203.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training
{
services::Status checkImpl(const decision_forest::training::interface2::Parameter & prm);
}

namespace classification
{
namespace training
{
namespace interface1
{
class Result::ResultImpl
{
public:
    ResultImpl() {}
    ResultImpl(const ResultImpl & other)
    {
        _engine = other._engine;
    }

    void setEngine(engines::engine_type engine) { _engine = engine; }
    engines::engine_type getEngine()
    {
        if (_engine == engines::engine_type::mt2203) _engine = engines::engine_type::mt2203;
        return _engine;
    }

private:
    engines::engine_type _engine;
};

} // namespace interface1
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
