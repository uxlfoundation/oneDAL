/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

class base_homogen_table_serialization_test : public te::policy_fixture {
public:
    virtual void compare_tables(const homogen_table& original, const table& deserialized) = 0;

    virtual void compare_tables(const homogen_table& original,
                                const homogen_table& deserialized) = 0;

    void check_table_serialization(const homogen_table& original) {
        SECTION("deserialize as exact type") {
            homogen_table deserialized;

            SECTION("serialize as exact type") {
                te::serialize_deserialize(original, deserialized);
            }

            SECTION("serialize as base type") {
                table original_as_base = original;
                te::serialize_deserialize(original_as_base, deserialized);
            }

            compare_tables(original, deserialized);
        }

        SECTION("deserialize as base type") {
            table deserialized;

            SECTION("serialize as exact type") {
                te::serialize_deserialize(original, deserialized);
            }

            SECTION("serialize as base type") {
                table original_as_base = original;
                te::serialize_deserialize(original_as_base, deserialized);
            }

            compare_tables(original, deserialized);
        }
    }
};

class empty_homogen_table_serialization_test : public base_homogen_table_serialization_test {
public:
    void compare_tables(const homogen_table& original, const table& deserialized) override {
        check_empty_base_table(deserialized);
    }

    void compare_tables(const homogen_table& original, const homogen_table& deserialized) override {
        check_empty_homogen_table(deserialized);
    }

    void check_empty_base_table(const table& deserialized) {
        REQUIRE(deserialized.get_row_count() == 0);
        REQUIRE(deserialized.get_column_count() == 0);
        REQUIRE(deserialized.has_data() == false);
        REQUIRE(deserialized.get_data_layout() == data_layout::unknown);
        REQUIRE(deserialized.get_metadata().get_feature_count() == 0);
    }

    void check_empty_homogen_table(const homogen_table& deserialized) {
        check_empty_base_table(deserialized);
        REQUIRE(deserialized.kind() == homogen_table::kind());
        REQUIRE(deserialized.get_data() == nullptr);
    }
};

template <typename Data>
class homogen_table_serialization_test : public base_homogen_table_serialization_test {
public:
    homogen_table get_host_backed_table(std::int64_t row_count,
                                        std::int64_t column_count,
                                        int seed = 7777) {
        auto random_mat =
            la::generate_uniform_matrix<Data>({ row_count, column_count }, -10, 10, seed);
        return homogen_table::wrap(random_mat.get_array(), row_count, column_count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    homogen_table get_device_backed_table(std::int64_t row_count,
                                          std::int64_t column_count,
                                          int seed = 7777) {
        auto random_mat =
            la::generate_uniform_matrix<Data>({ row_count, column_count }, -10, 10, seed);
        auto random_device_ary = random_mat.to_device(this->get_queue()).get_array();
        return homogen_table::wrap(random_device_ary, row_count, column_count);
    }
#endif

    void compare_tables(const homogen_table& original, const table& deserialized) override {
        te::check_if_tables_equal<Data>(deserialized, original);
        // The table is deserialized without a queue (see the serialization test
        // engine), so its data is materialized on the host regardless of where
        // the original resided. The allocation kind is therefore non_usm.
        REQUIRE(deserialized.get_metadata().get_alloc_kind() == alloc_kind::non_usm);
    }

    void compare_tables(const homogen_table& original, const homogen_table& deserialized) override {
        te::check_if_tables_equal<Data>(deserialized, original);
        REQUIRE(deserialized.get_metadata().get_alloc_kind() == alloc_kind::non_usm);
    }
};

TEST_M(empty_homogen_table_serialization_test, "serialize/deserialize empty homogen table") {
    const homogen_table empty_table;

    check_table_serialization(empty_table);
}

using homogen_table_types = COMBINE_TYPES((float, double, std::int32_t));

TEMPLATE_LIST_TEST_M(homogen_table_serialization_test,
                     "serialize/deserialize host homogen table",
                     "[host]",
                     homogen_table_types) {
    const std::int64_t row_count = GENERATE(1, 10, 1000);
    const std::int64_t column_count = GENERATE(1, 10, 100);
    const homogen_table original = this->get_host_backed_table(row_count, column_count);

    this->check_table_serialization(original);
}

#ifdef ONEDAL_DATA_PARALLEL
TEMPLATE_LIST_TEST_M(homogen_table_serialization_test,
                     "serialize/deserialize device homogen table",
                     "[host]",
                     homogen_table_types) {
    const std::int64_t row_count = GENERATE(1, 10, 1000);
    const std::int64_t column_count = GENERATE(1, 10, 100);
    const homogen_table original = this->get_device_backed_table(row_count, column_count);

    this->check_table_serialization(original);
}
#endif

} // namespace oneapi::dal::test
