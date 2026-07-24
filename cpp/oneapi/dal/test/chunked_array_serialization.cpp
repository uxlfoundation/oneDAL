/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/chunked_array.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename T>
class chunked_array_serialization_test : te::policy_fixture {
public:
    static array<T> get_empty_array() {
        return array<T>{};
    }

    static chunked_array<T> get_empty_chunked_array() {
        return chunked_array<T>{};
    }

    array<T> get_host_backed_array(std::int64_t count, int seed = 7777) {
        auto random_mat = la::generate_uniform_matrix<T>({ 1, count }, -10, 10, seed);
        return random_mat.get_array();
    }

#ifdef ONEDAL_DATA_PARALLEL
    array<T> get_device_backed_array(std::int64_t count, int seed = 7777) {
        auto random_mat = la::generate_uniform_matrix<T>({ 1, count }, -10, 10, seed);
        return random_mat.to_device(this->get_queue()).get_array();
    }
#endif

    void compare_arrays(const chunked_array<T>& original, const chunked_array<T>& deserialized) {
        REQUIRE(original.get_count() == deserialized.get_count());

        if (original.get_count() > 0) {
            const auto original_flatten = original.flatten();
            const auto deserialized_flatten = deserialized.flatten();

            const auto original_host = la::matrix<T>::wrap(original_flatten) //
                                           .to_host()
                                           .get_array();
            const auto deserialized_host = la::matrix<T>::wrap(deserialized_flatten) //
                                               .to_host()
                                               .get_array();

            for (std::int64_t i = 0; i < original.get_count(); ++i) {
                REQUIRE(deserialized_host[i] == original_host[i]);
            }
        }
    }
};

using array_types = std::tuple<std::int32_t, float, double>;

TEMPLATE_LIST_TEST_M(chunked_array_serialization_test,
                     "serialize/deserialize empty chunked_array",
                     "[empty]",
                     array_types) {
    const auto empty_chunked_array = this->get_empty_chunked_array();

    const auto deserialized = te::serialize_deserialize(empty_chunked_array);

    this->compare_arrays(empty_chunked_array, deserialized);
}

TEMPLATE_LIST_TEST_M(chunked_array_serialization_test,
                     "serialize/deserialize two host arrays",
                     "[host]",
                     array_types) {
    const std::int64_t host_count1 = GENERATE(1, 10, 1071);
    const std::int64_t host_count2 = GENERATE(5, 11, 1023);
    array<TestType> host_array1 = this->get_host_backed_array(host_count1, host_count1);
    array<TestType> host_array2 = this->get_host_backed_array(host_count2, host_count2);

    chunked_array<TestType> host_chunked;

    host_chunked.append(host_array1, host_array2, host_array1);

    const auto deserialized = te::serialize_deserialize(host_chunked);

    this->compare_arrays(host_chunked, deserialized);
}

#ifdef ONEDAL_DATA_PARALLEL
TEMPLATE_LIST_TEST_M(chunked_array_serialization_test,
                     "serialize/deserialize device arrays",
                     "[device]",
                     array_types) {
    const std::int64_t arr1_count = GENERATE(3, 7, 1027);
    const std::int64_t arr2_count = GENERATE(5, 11, 1023);
    array<TestType> arr1 = this->get_device_backed_array(arr1_count, arr1_count);
    array<TestType> arr2 = this->get_device_backed_array(arr2_count, arr2_count);

    chunked_array<TestType> chunked;

    chunked.append(arr1, arr2, arr1);

    const auto deserialized = te::serialize_deserialize(chunked);

    this->compare_arrays(chunked, deserialized);
}
#endif

} // namespace oneapi::dal::test
