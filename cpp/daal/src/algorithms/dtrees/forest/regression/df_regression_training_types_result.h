#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "src/algorithms/engines/engine.h"
#include "src/algorithms/engines/engine_factory.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
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

    void setEngine(engines::EnginePtr engine)
    {
        _engine = engine;
    }

    engines::EnginePtr getEngine()
    {
        return _engine;
    }

private:
    engines::EnginePtr _engine;
};

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
