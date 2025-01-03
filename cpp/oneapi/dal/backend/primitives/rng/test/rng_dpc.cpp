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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/backend/primitives/rng/rng.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine_collection.hpp"
namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

class mt2203 {};
class mcg59 {};
class mrg32k3a {};
class mt19937 {};
class philox4x32x10 {};

template <typename engine_type>
struct engine_map {};

template <>
struct engine_map<mt2203> {
    constexpr static auto value = engine_method::mt2203;
};

template <>
struct engine_map<mcg59> {
    constexpr static auto value = engine_method::mcg59;
};

template <>
struct engine_map<mrg32k3a> {
    constexpr static auto value = engine_method::mrg32k3a;
};

template <>
struct engine_map<philox4x32x10> {
    constexpr static auto value = engine_method::philox4x32x10;
};

template <>
struct engine_map<mt19937> {
    constexpr static auto value = engine_method::mt19937;
};

template <typename engine_type>
constexpr auto engine_v = engine_map<engine_type>::value;

template <typename TestType>
class rng_test : public te::policy_fixture {
public:
    using DataType = std::tuple_element_t<0, TestType>;
    using EngineType = std::tuple_element_t<1, TestType>;
    static constexpr auto engine_test_type = engine_v<EngineType>;

    auto get_host_engine(std::int64_t seed) {
        auto rng_engine = host_engine<engine_test_type>(seed);
        return rng_engine;
    }

    auto get_dpc_engine(std::int64_t seed) {
        auto rng_engine = dpc_engine<engine_test_type>(this->get_queue(), seed);
        return rng_engine;
    }

    auto allocate_array_host(std::int64_t elem_count) {
        auto arr_host = ndarray<DataType, 1>::empty({ elem_count });
        return arr_host;
    }

    auto allocate_array_device(std::int64_t elem_count) {
        auto& q = this->get_queue();
        auto arr_gpu = ndarray<DataType, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        return arr_gpu;
    }

    void check_results(const ndarray<DataType, 1>& arr_1, const ndarray<DataType, 1>& arr_2) {
        const auto arr_1_host = arr_1.to_host(this->get_queue());
        const DataType* val_arr_1_host_ptr = arr_1_host.get_data();

        const auto arr_2_host = arr_2.to_host(this->get_queue());
        const DataType* val_arr_2_host_ptr = arr_2_host.get_data();

        for (std::int64_t el = 0; el < arr_2_host.get_count(); el++) {
            // Due to MKL inside generates floats on GPU and doubles on CPU, it makes sense to add minor eps.
            REQUIRE(abs(val_arr_1_host_ptr[el] - val_arr_2_host_ptr[el]) < 0.1);
        }
    }
};

using rng_types = COMBINE_TYPES((float, double), (mt2203, mt19937, mcg59, mrg32k3a, philox4x32x10));

TEMPLATE_LIST_TEST_M(rng_test, "rng cpu vs gpu", "[rng]", rng_types) {
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    std::int64_t elem_count = GENERATE_COPY(10, 777, 10000, 50000);
    std::int64_t seed = GENERATE_COPY(777, 999);

    auto arr_gpu = this->allocate_array_device(elem_count);
    auto arr_host = this->allocate_array_host(elem_count);
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rng_engine = this->get_dpc_engine(seed);
    auto rng_engine_ = this->get_dpc_engine(seed);

    uniform<Float>(elem_count, arr_host_ptr, rng_engine, 0, elem_count);
    uniform<Float>(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine_, 0, elem_count);

    this->check_results(arr_gpu, arr_host);
}

using rng_types_skip_ahead_support = COMBINE_TYPES((float, double),
                                                   (mt19937, mcg59, mrg32k3a, philox4x32x10));

TEMPLATE_LIST_TEST_M(rng_test, "mixed rng cpu skip", "[rng]", rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    std::int64_t elem_count = GENERATE_COPY(10, 777, 10000, 100000);
    std::int64_t seed = GENERATE_COPY(777, 999);

    auto arr_host_init_1 = this->allocate_array_host(elem_count);
    auto arr_host_init_2 = this->allocate_array_host(elem_count);

    auto arr_gpu = this->allocate_array_device(elem_count);
    auto arr_host = this->allocate_array_host(elem_count);

    auto arr_host_init_1_ptr = arr_host_init_1.get_mutable_data();
    auto arr_host_init_2_ptr = arr_host_init_2.get_mutable_data();
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rng_engine = this->get_dpc_engine(seed);
    auto rng_engine_2 = this->get_dpc_engine(seed);

    uniform<Float>(elem_count, arr_host_init_1_ptr, rng_engine, 0, elem_count);
    uniform<Float>(elem_count, arr_host_init_2_ptr, rng_engine_2, 0, elem_count);

    uniform<Float>(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine, 0, elem_count);
    uniform<Float>(elem_count, arr_host_ptr, rng_engine_2, 0, elem_count);

    this->check_results(arr_host_init_1, arr_host_init_2);
    this->check_results(arr_gpu, arr_host);
}

TEMPLATE_LIST_TEST_M(rng_test, "mixed rng gpu skip", "[rng]", rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    std::int64_t elem_count = GENERATE_COPY(10, 100, 777, 10000);
    std::int64_t seed = GENERATE_COPY(1, 777, 999);

    auto arr_device_init_1 = this->allocate_array_device(elem_count);
    auto arr_device_init_2 = this->allocate_array_device(elem_count);

    auto arr_gpu = this->allocate_array_device(elem_count);
    auto arr_host = this->allocate_array_host(elem_count);

    auto arr_device_init_1_ptr = arr_device_init_1.get_mutable_data();
    auto arr_device_init_2_ptr = arr_device_init_2.get_mutable_data();
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rng_engine = this->get_dpc_engine(seed);
    auto rng_engine_2 = this->get_dpc_engine(seed);

    uniform<Float>(this->get_queue(), elem_count, arr_device_init_1_ptr, rng_engine, 0, elem_count);
    uniform<Float>(this->get_queue(),
                   elem_count,
                   arr_device_init_2_ptr,
                   rng_engine_2,
                   0,
                   elem_count);

    uniform<Float>(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine, 0, elem_count);
    uniform<Float>(elem_count, arr_host_ptr, rng_engine_2, 0, elem_count);

    this->check_results(arr_device_init_1, arr_device_init_2);
    this->check_results(arr_gpu, arr_host);
}

//TODO: add engine collection test + host_engine tests
// TEMPLATE_LIST_TEST_M(rng_test, "mixed rng gpu skip collection", "[rng]", rng_types_skip) {
//     SKIP_IF(this->get_policy().is_cpu());
//     std::int64_t elem_count = GENERATE_COPY(10, 100, 777, 10000);
//     std::int64_t seed = GENERATE_COPY(1, 777, 999);

//     engine_collection<std::int64_t,engine_method::mcg59> collection(this->get_queue(), 2, seed);

//     auto engine_arr = collection.get_dpc_engines();

//     auto [arr_device_init_1, arr_device_init_2] = this->allocate_arrays_shared(elem_count);

//     auto arr_device_init_1_ptr = arr_device_init_1.get_mutable_data();
//     auto arr_device_init_2_ptr = arr_device_init_2.get_mutable_data();

//     auto rn_gen = this->get_rng();

//     rn_gen.uniform(this->get_queue(),
//                    elem_count,
//                    arr_device_init_1_ptr,
//                    engine_arr[0],
//                    0,
//                    elem_count);

//     rn_gen.uniform(this->get_queue(),
//                    elem_count,
//                    arr_device_init_2_ptr,
//                    engine_arr[1],
//                    0,
//                    elem_count);

//     // rn_gen.uniform(this->get_queue(), elem_count, arr_gpu_ptr, engine_arr[0], 0, elem_count);
//     // rn_gen.uniform(elem_count, arr_host_ptr, engine_arr[1], 0, elem_count);

//     //this->check_results_device(arr_device_init_1, arr_device_init_2);
//     this->check_results(arr_device_init_1, arr_device_init_2);
// }

} // namespace oneapi::dal::backend::primitives::test
