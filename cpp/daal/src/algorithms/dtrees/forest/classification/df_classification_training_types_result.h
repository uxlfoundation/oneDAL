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
        if (other._engine) _engine = other._engine->clone();
    }

    void setEngine(engines::EnginePtr engine) { _engine = engine; }
    engines::EnginePtr getEngine()
    {
        if (!_engine) _engine = engines::mt2203::Batch<>::create();
        return _engine;
    }

private:
    engines::EnginePtr _engine;
};

} // namespace interface1
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
