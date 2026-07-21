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

#include "oneapi/dal/table/detail/metadata_utils.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::detail {

void check_dtype_correctness(const data_type* dtypes, const table_metadata& meta) {
    const auto column_count = meta.get_feature_count();

    for (std::int64_t c = 0l; c < column_count; ++c) {
        REQUIRE(dtypes[c] == meta.get_data_type(c));
    }
}

void check_ftype_correctness(const feature_type* ftypes, const table_metadata& meta) {
    const auto column_count = meta.get_feature_count();

    for (std::int64_t c = 0l; c < column_count; ++c) {
        REQUIRE(ftypes[c] == meta.get_feature_type(c));
    }
}

constexpr data_type expected_dtypes[] = { //
    data_type::int32,
    data_type::int64,
    data_type::float32,
    data_type::float64
};
constexpr feature_type expected_ftypes[] = { //
    feature_type::ordinal,
    feature_type::ordinal,
    feature_type::ratio,
    feature_type::ratio
};

TEST("can get correcty metadata from raw types") {
    const auto meta = make_default_metadata<std::int32_t, std::int64_t, float, double>();

    check_dtype_correctness(expected_dtypes, meta);
    check_ftype_correctness(expected_ftypes, meta);
}

TEST("can get correcty metadata from array types") {
    const auto meta = make_default_metadata_from_arrays(dal::array<std::int32_t>{},
                                                        dal::array<std::int64_t>{},
                                                        dal::array<float>{},
                                                        dal::array<double>{});

    check_dtype_correctness(expected_dtypes, meta);
    check_ftype_correctness(expected_ftypes, meta);
    REQUIRE(meta.get_alloc_kind() == alloc_kind::non_usm);
}

} // namespace oneapi::dal::detail
