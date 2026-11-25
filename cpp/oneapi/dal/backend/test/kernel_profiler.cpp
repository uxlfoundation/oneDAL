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

#include <cstring>
#include <string>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::kmeans_like {
namespace method {
inline namespace v1 {
struct lloyd_dense {};
} // namespace v1
} // namespace method

namespace backend {
template <typename Float, typename Method>
struct train_kernel_cpu {
    int operator()(const dal::backend::context_cpu&, int x) const {
        return x + 1;
    }
};
} // namespace backend
} // namespace oneapi::dal::kmeans_like

namespace oneapi::dal::backend::test {

namespace kl = dal::kmeans_like;

struct interop_like_step {
    static constexpr const char* suffix = ".step";
};

static std::string shorten(const char* type) {
    return daal::internal::profiler_kernel_task_name(type, std::strlen(type), "");
}

TEST("kernel task name keeps the algorithm, the type and the last scope of its arguments") {
    REQUIRE(shorten("oneapi::dal::kmeans::backend::train_kernel_cpu<float, "
                    "oneapi::dal::kmeans::method::v1::lloyd_dense, "
                    "oneapi::dal::kmeans::task::v1::clustering>") ==
            "kmeans.train_kernel_cpu<float,lloyd_dense,clustering>");
    REQUIRE(shorten("daal::algorithms::decision_forest::classification::training::internal::"
                    "ClassificationTrainBatchKernel<float, "
                    "daal::algorithms::decision_forest::classification::training::defaultDense, "
                    "daal::internal::avx2>") ==
            "decision_forest.ClassificationTrainBatchKernel<float,defaultDense,avx2>");
    REQUIRE(shorten("oneapi::dal::pca::parameters::train_parameters_cpu<float>") ==
            "pca.train_parameters_cpu<float>");
}

TEST("kernel task name drops anonymous namespaces and the MSVC class keys") {
    REQUIRE(shorten("(anonymous namespace)::local_kernel<double>") == "local_kernel<double>");
    REQUIRE(shorten("{anonymous}::local_kernel<double>") == "local_kernel<double>");
    REQUIRE(shorten("struct oneapi::dal::svm::backend::infer_kernel_cpu<float,struct "
                    "oneapi::dal::svm::method::v1::thunder,struct `anonymous "
                    "namespace'::tag>") == "svm.infer_kernel_cpu<float,thunder,tag>");
}

TEST("kernel task name of a type keeps its address and carries the step") {
    using kernel_t = kl::backend::train_kernel_cpu<double, kl::method::lloyd_dense>;
    const char* const name = kernel_task_name<kernel_t>();
    REQUIRE(std::string(name) == "kmeans_like.train_kernel_cpu<double,lloyd_dense>");
    REQUIRE(kernel_task_name<kernel_t>() == name);
    REQUIRE(std::string(kernel_task_name<kernel_t, interop_like_step>()) ==
            "kmeans_like.train_kernel_cpu<double,lloyd_dense>.step");
}

TEST("kernel dispatcher runs the kernel with the profiler off") {
    using kernel_t = kl::backend::train_kernel_cpu<float, kl::method::lloyd_dense>;
    const kernel_dispatcher<kernel_spec<single_node_cpu_kernel, kernel_t>> dispatch;
    REQUIRE(dispatch(detail::host_policy::get_default(), 41) == 42);
}

} // namespace oneapi::dal::backend::test
