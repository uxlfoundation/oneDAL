#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"

namespace oneapi::dal::linear_regression::backend {

namespace daal_lr = daal::algorithms::linear_regression;
using daal_lr_hyperparameters_t = daal_lr::internal::Hyperparameter;
namespace interop = dal::backend::interop;

template <typename Float, typename Task>
static daal_lr_hyperparameters_t convert_parameters(const detail::train_parameters<Task>& params) {
    using daal_lr::internal::HyperparameterId;

    const std::int64_t block = params.get_cpu_macro_block();
    const std::int64_t max_cols_batched = params.get_cpu_max_cols_batched();
    const std::int64_t small_rows_threshold = params.get_cpu_small_rows_threshold();
    const std::int64_t small_rows_max_cols_batched = params.get_cpu_small_rows_max_cols_batched();

    daal_lr_hyperparameters_t daal_hyperparameter;
    auto status = daal_hyperparameter.set(HyperparameterId::denseUpdateStepBlockSize, block);
    interop::status_to_exception(status);
    status = daal_hyperparameter.set(HyperparameterId::denseUpdateMaxColsBatched, max_cols_batched);
    interop::status_to_exception(status);
    status =
        daal_hyperparameter.set(HyperparameterId::denseSmallRowsThreshold, small_rows_threshold);
    interop::status_to_exception(status);
    status = daal_hyperparameter.set(HyperparameterId::denseSmallRowsMaxColsBatched,
                                     small_rows_max_cols_batched);
    interop::status_to_exception(status);

    return daal_hyperparameter;
}

} // namespace oneapi::dal::linear_regression::backend
