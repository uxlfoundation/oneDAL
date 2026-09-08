/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <array>

#include "oneapi/dal/algo/basic_statistics/compute.hpp"
#include "oneapi/dal/algo/basic_statistics/partial_compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;

template <typename TestType>
class bs_badarg_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor() const {
        return basic_statistics::descriptor<Float, Method>{};
    }

#ifdef ONEDAL_DATA_PARALLEL
    static constexpr std::int64_t block_count_ = 2;
    static constexpr std::int64_t rows_in_block_ = 2;
    static constexpr std::int64_t row_count_ = block_count_ * rows_in_block_;
    static constexpr std::int64_t column_count_ = 2;

    /// Returns the block of `data_` with the given index, allocated in the memory of
    /// the requested kind on the queue the computations are performed on.
    table get_block(std::int64_t block_index, alloc_kind alloc) {
        const auto block = te::get_table_block<Float>(this->get_policy(),
                                                      get_data(),
                                                      get_rows(block_index),
                                                      alloc);
        return homogen_table::wrap(block, rows_in_block_, column_count_);
    }

    /// Returns the block of `data_` with the given index, allocated in device USM
    /// memory on the given queue.
    table get_block_on_queue(std::int64_t block_index, sycl::queue& queue) {
        const auto block = row_accessor<const Float>{ get_data() }.pull(queue,
                                                                        get_rows(block_index),
                                                                        sycl::usm::alloc::device);
        return homogen_table::wrap(block, rows_in_block_, column_count_);
    }

    /// Returns the weights for a single block of `data_`, allocated in device USM
    /// memory on the given queue.
    table get_weights_on_queue(sycl::queue& queue) {
        auto weights = array<Float>::empty(queue, rows_in_block_, sycl::usm::alloc::device);
        queue.fill(weights.get_mutable_data(), Float(1), rows_in_block_).wait_and_throw();
        return homogen_table::wrap(weights, rows_in_block_, 1);
    }

    /// Returns a queue that differs from the one the computations are performed on,
    /// but shares the context with it, so that only the queues do not match.
    sycl::queue get_other_queue() {
        auto& queue = this->get_queue();
        return sycl::queue{ queue.get_context(), queue.get_device() };
    }

    table get_data() const {
        return homogen_table::wrap(data_.data(), row_count_, column_count_);
    }

private:
    range get_rows(std::int64_t block_index) const {
        const std::int64_t row_offset = block_index * rows_in_block_;
        return range{ row_offset, row_offset + rows_in_block_ };
    }

    std::array<Float, row_count_ * column_count_> data_ = { 1, 2, 3, 4, 5, 6, 7, 8 };
#endif
};

using bs_types = COMBINE_TYPES((float, double), (basic_statistics::method::dense));

#define BS_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(bs_badarg_test, name, "[basic_statistics_kernel][badarg]", bs_types)

BS_BADARG_TEST("throws if data is empty") {
    const auto desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(desc, homogen_table{}), domain_error);
}

#ifdef ONEDAL_DATA_PARALLEL

BS_BADARG_TEST("throws if the next block is not allocated in USM while the first one is") {
    SKIP_IF(this->not_float64_friendly());
    const auto desc = this->get_descriptor();

    const auto partial_result = this->partial_compute(desc,
                                                      partial_compute_result<>{},
                                                      this->get_block(0, alloc_kind::usm_device));

    REQUIRE_THROWS_AS(
        this->partial_compute(desc, partial_result, this->get_block(1, alloc_kind::non_usm)),
        domain_error);
}

BS_BADARG_TEST("throws if the next block is allocated in USM while the first one is not") {
    SKIP_IF(this->not_float64_friendly());
    const auto desc = this->get_descriptor();

    const auto partial_result = this->partial_compute(desc,
                                                      partial_compute_result<>{},
                                                      this->get_block(0, alloc_kind::non_usm));

    REQUIRE_THROWS_AS(
        this->partial_compute(desc, partial_result, this->get_block(1, alloc_kind::usm_device)),
        domain_error);
}

BS_BADARG_TEST("throws if the next block comes from another queue") {
    SKIP_IF(this->not_float64_friendly());
    const auto desc = this->get_descriptor();

    const auto partial_result = this->partial_compute(desc,
                                                      partial_compute_result<>{},
                                                      this->get_block(0, alloc_kind::usm_device));

    auto other_queue = this->get_other_queue();
    REQUIRE_THROWS_AS(
        this->partial_compute(desc, partial_result, this->get_block_on_queue(1, other_queue)),
        domain_error);
}

BS_BADARG_TEST("throws if weights come from another queue than data") {
    SKIP_IF(this->not_float64_friendly());
    const auto desc = this->get_descriptor();

    auto other_queue = this->get_other_queue();
    const auto data = this->get_block(0, alloc_kind::usm_device);
    const auto weights = this->get_weights_on_queue(other_queue);

    REQUIRE_THROWS_AS(this->partial_compute(desc, partial_compute_result<>{}, data, weights),
                      domain_error);
}

#endif

} // namespace oneapi::dal::basic_statistics::test
