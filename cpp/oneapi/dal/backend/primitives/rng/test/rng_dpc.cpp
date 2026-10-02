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

#include "oneapi/dal/backend/primitives/rng/device_engine.hpp"
#include "oneapi/dal/backend/primitives/rng/host_engine.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_types.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include <vector>

#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

class mt2203 {};
class mcg59 {};
class mrg32k3a {};
class mt19937 {};
class philox4x32x10 {};

template <typename engine_type_internal>
struct engine_map {};

template <>
struct engine_map<mt2203> {
    constexpr static auto value = engine_type_internal::mt2203;
};

template <>
struct engine_map<mcg59> {
    constexpr static auto value = engine_type_internal::mcg59;
};

template <>
struct engine_map<mrg32k3a> {
    constexpr static auto value = engine_type_internal::mrg32k3a;
};

template <>
struct engine_map<philox4x32x10> {
    constexpr static auto value = engine_type_internal::philox4x32x10;
};

template <>
struct engine_map<mt19937> {
    constexpr static auto value = engine_type_internal::mt19937;
};

template <typename engine_type_internal>
constexpr auto engine_v = engine_map<engine_type_internal>::value;

template <typename TestType>
class rng_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using DataType = std::tuple_element_t<0, TestType>;
    using EngineType = std::tuple_element_t<1, TestType>;
    static constexpr auto engine_test_type = engine_v<EngineType>;

    auto get_host_engine(std::int64_t seed) {
        auto rng_engine = host_engine(seed, engine_test_type);
        return rng_engine;
    }

    auto get_device_engine(std::int64_t seed) {
        auto rng_engine = device_engine(this->get_queue(), seed, engine_test_type);
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
            REQUIRE(abs(val_arr_1_host_ptr[el] - val_arr_2_host_ptr[el]) < 0.01);
        }
    }

    /// Checks that `tail` reproduces `full` starting at `offset`.
    ///
    /// @param[in] full   The reference values, at least `offset + tail.get_count()` of them.
    /// @param[in] tail   The values drawn after the stream was skipped by `offset`.
    /// @param[in] offset The number of values the skipped stream jumped over.
    void check_tail_matches(const ndarray<DataType, 1>& full,
                            const ndarray<DataType, 1>& tail,
                            std::int64_t offset) {
        const auto full_host = full.to_host(this->get_queue());
        const auto tail_host = tail.to_host(this->get_queue());
        REQUIRE(full_host.get_count() >= offset + tail_host.get_count());
        for (std::int64_t el = 0; el < tail_host.get_count(); el++) {
            REQUIRE(full_host.get_data()[offset + el] == tail_host.get_data()[el]);
        }
    }

    auto allocate_indices(std::int64_t elem_count) {
        return ndarray<std::int32_t, 1>::empty(this->get_queue(),
                                               { elem_count },
                                               sycl::usm::alloc::host);
    }

    /// Draws `count` values from the device stream of `engine_`.
    ///
    /// @param[in] count   The number of values to draw.
    /// @param[in] engine_ The engine whose device stream is advanced.
    /// @return A device `ndarray` holding the `count` drawn values.
    auto draw_on_device(std::int64_t count, device_engine& engine_) {
        auto arr = this->allocate_array_device(count);
        uniform<DataType>(this->get_queue(),
                          count,
                          arr.get_mutable_data(),
                          engine_,
                          DataType(0),
                          DataType(1))
            .wait_and_throw();
        return arr;
    }

    /// Draws `count` values from the host stream of `engine_`.
    ///
    /// @param[in] count   The number of values to draw.
    /// @param[in] engine_ The engine whose host stream is advanced.
    /// @return A host `ndarray` holding the `count` drawn values.
    auto draw_on_host(std::int64_t count, device_engine& engine_) {
        auto arr = this->allocate_array_host(count);
        uniform<DataType>(count, arr.get_mutable_data(), engine_, DataType(0), DataType(1));
        return arr;
    }
};

using rng_types = COMBINE_TYPES((float, double), (mt2203, mt19937, mcg59, mrg32k3a, philox4x32x10));

TEMPLATE_LIST_TEST_M(rng_test, "rng cpu vs gpu", "[rng]", rng_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    std::int64_t elem_count = GENERATE_COPY(10, 777, 10000, 50000);
    std::int64_t seed = GENERATE_COPY(777, 999);

    auto arr_gpu = this->allocate_array_device(elem_count);
    auto arr_host = this->allocate_array_host(elem_count);
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rng_engine = this->get_device_engine(seed);
    auto rng_engine_ = this->get_device_engine(seed);

    uniform<Float>(elem_count, arr_host_ptr, rng_engine, 0, elem_count);
    uniform<Float>(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine_, 0, elem_count)
        .wait_and_throw();

    this->check_results(arr_gpu, arr_host);
}

using rng_types_skip_ahead_support = COMBINE_TYPES((float, double),
                                                   (mt19937, mcg59, mrg32k3a, philox4x32x10));

TEMPLATE_LIST_TEST_M(rng_test, "mixed rng cpu skip", "[rng]", rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
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

    auto rng_engine = this->get_device_engine(seed);
    auto rng_engine_2 = this->get_device_engine(seed);

    uniform<Float>(elem_count, arr_host_init_1_ptr, rng_engine, 0, elem_count);
    uniform<Float>(elem_count, arr_host_init_2_ptr, rng_engine_2, 0, elem_count);

    uniform<Float>(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine, 0, elem_count)
        .wait_and_throw();
    uniform<Float>(elem_count, arr_host_ptr, rng_engine_2, 0, elem_count);

    this->check_results(arr_host_init_1, arr_host_init_2);
    this->check_results(arr_gpu, arr_host);
}

TEMPLATE_LIST_TEST_M(rng_test, "mixed rng gpu skip", "[rng]", rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
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

    auto rng_engine = this->get_device_engine(seed);
    auto rng_engine_2 = this->get_device_engine(seed);

    uniform<Float>(this->get_queue(), elem_count, arr_device_init_1_ptr, rng_engine, 0, elem_count)
        .wait_and_throw();
    uniform<Float>(this->get_queue(),
                   elem_count,
                   arr_device_init_2_ptr,
                   rng_engine_2,
                   0,
                   elem_count)
        .wait_and_throw();

    uniform<Float>(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine, 0, elem_count)
        .wait_and_throw();
    uniform<Float>(elem_count, arr_host_ptr, rng_engine_2, 0, elem_count);

    this->check_results(arr_device_init_1, arr_device_init_2);
    this->check_results(arr_gpu, arr_host);
}

using rng_types_default_engine = COMBINE_TYPES((float), (philox4x32x10));

TEMPLATE_LIST_TEST_M(rng_test, "default engine and seed", "[rng]", rng_types_default_engine) {
    SKIP_IF(this->get_policy().is_cpu());
    using Float = std::tuple_element_t<0, TestType>;

    auto& q = this->get_queue();
    auto defaulted = device_engine(q);
    auto explicit_ = device_engine(q, default_seed, default_engine_type_internal);

    REQUIRE(defaulted.get_device_engine_base_ptr()->get_engine_type() ==
            engine_type_internal::philox4x32x10);

    constexpr std::int64_t count = 128;
    auto arr_defaulted = this->allocate_array_device(count);
    auto arr_explicit = this->allocate_array_device(count);
    uniform<Float>(q, count, arr_defaulted.get_mutable_data(), defaulted, 0, 1).wait_and_throw();
    uniform<Float>(q, count, arr_explicit.get_mutable_data(), explicit_, 0, 1).wait_and_throw();
    this->check_tail_matches(arr_defaulted, arr_explicit, 0);
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "skip ahead offsets host and device streams",
                     "[rng]",
                     rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t skip = GENERATE_COPY(1, 128, 10000);
    constexpr std::int64_t count = 256;
    CAPTURE(skip, count);

    auto dev_full = this->get_device_engine(default_seed);
    auto dev_tail = this->get_device_engine(default_seed);
    dev_tail.skip_ahead(skip);
    this->check_tail_matches(this->draw_on_device(skip + count, dev_full),
                             this->draw_on_device(count, dev_tail),
                             skip);

    // A draw mirrors its own length onto the other stream, and the two draws above have different
    // lengths, so the host check needs a pair of engines the device check has not advanced.
    auto host_full = this->get_device_engine(default_seed);
    auto host_tail = this->get_device_engine(default_seed);
    host_tail.skip_ahead(skip);
    this->check_tail_matches(this->draw_on_host(skip + count, host_full),
                             this->draw_on_host(count, host_tail),
                             skip);
}

using rng_types_no_skip_ahead = COMBINE_TYPES((float), (mt2203));

TEMPLATE_LIST_TEST_M(rng_test,
                     "mt2203 device skip ahead is a no-op",
                     "[rng]",
                     rng_types_no_skip_ahead) {
    SKIP_IF(this->get_policy().is_cpu());

    constexpr std::int64_t count = 256;
    auto eng_plain = this->get_device_engine(default_seed);
    auto eng_skipped = this->get_device_engine(default_seed);
    eng_skipped.skip_ahead_gpu(count);

    // Documented on `gen_mt2203::skip_ahead_gpu`: oneMKL has no `skip_ahead` for mt2203, so the
    // skip-based stream separation of distributed decision forest cannot work with this engine.
    this->check_tail_matches(this->draw_on_device(count, eng_plain),
                             this->draw_on_device(count, eng_skipped),
                             0);
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "shuffle keeps host and device streams in lockstep",
                     "[rng]",
                     rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t count = GENERATE_COPY(1, 10, 1000);
    constexpr std::int64_t probe_count = 128;

    auto eng_shuffled = this->get_device_engine(default_seed);
    auto eng_drawn = this->get_device_engine(default_seed);

    auto order = this->allocate_indices(count);
    shuffle<std::int32_t>(count, order.get_mutable_data(), eng_shuffled);
    // `shuffle` draws two values per iteration, so an equivalent plain draw is `2 * count` long.
    auto drawn = this->allocate_indices(2 * count);
    uniform<std::int32_t>(2 * count,
                          drawn.get_mutable_data(),
                          eng_drawn,
                          0,
                          static_cast<std::int32_t>(count));

    this->check_tail_matches(this->draw_on_device(probe_count, eng_shuffled),
                             this->draw_on_device(probe_count, eng_drawn),
                             0);
    this->check_tail_matches(this->draw_on_host(probe_count, eng_shuffled),
                             this->draw_on_host(probe_count, eng_drawn),
                             0);
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "uniform without replacement keeps host and device streams in lockstep",
                     "[rng]",
                     rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t count = GENERATE_COPY(1, 10, 1000);
    constexpr std::int32_t top = 4096;
    constexpr std::int64_t probe_count = 128;

    auto eng_sampled = this->get_device_engine(default_seed);
    auto eng_drawn = this->get_device_engine(default_seed);

    auto sample = this->allocate_indices(count);
    uniform_without_replacement<std::int32_t>(count,
                                              sample.get_mutable_data(),
                                              eng_sampled,
                                              0,
                                              top);
    // One value is drawn per output element, so an equivalent plain draw is `count` long.
    auto drawn = this->allocate_indices(count);
    uniform<std::int32_t>(count, drawn.get_mutable_data(), eng_drawn, 0, top);

    this->check_tail_matches(this->draw_on_device(probe_count, eng_sampled),
                             this->draw_on_device(probe_count, eng_drawn),
                             0);
    this->check_tail_matches(this->draw_on_host(probe_count, eng_sampled),
                             this->draw_on_host(probe_count, eng_drawn),
                             0);
}

// Decision forest picks the features of a node with this routine, so a repeated index would make
// a node split twice on the same feature.
TEMPLATE_LIST_TEST_M(rng_test,
                     "uniform without replacement draws distinct indices",
                     "[rng]",
                     rng_types_default_engine) {
    SKIP_IF(this->get_policy().is_cpu());

    const std::int32_t top = GENERATE_COPY(16, 1024);
    const std::int64_t count = GENERATE_COPY(1, top / 2, top);

    auto engine_ = this->get_device_engine(default_seed);
    auto sample = this->allocate_indices(count);
    uniform_without_replacement<std::int32_t>(count, sample.get_mutable_data(), engine_, 0, top);

    std::vector<bool> seen(top, false);
    for (std::int64_t i = 0; i < count; ++i) {
        const std::int32_t value = sample.get_data()[i];
        REQUIRE(value >= 0);
        REQUIRE(value < top);
        REQUIRE(!seen[value]);
        seen[value] = true;
    }
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "partial fisher yates keeps host and device streams in lockstep",
                     "[rng]",
                     rng_types_skip_ahead_support) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t count = GENERATE_COPY(1, 10, 1000);
    const std::int64_t top = 4096;
    constexpr std::int64_t probe_count = 128;

    auto& q = this->get_queue();
    auto eng_sampled = this->get_device_engine(default_seed);
    auto eng_drawn = this->get_device_engine(default_seed);

    auto sample = this->allocate_indices(count);
    partial_fisher_yates_shuffle<std::int32_t>(q, sample, top, eng_sampled).wait_and_throw();
    // One value is drawn per sampled index, so an equivalent plain draw is `count` long.
    std::vector<std::size_t> drawn(count);
    uniform<std::size_t>(count, drawn.data(), eng_drawn, 0, static_cast<std::size_t>(top));

    this->check_tail_matches(this->draw_on_device(probe_count, eng_sampled),
                             this->draw_on_device(probe_count, eng_drawn),
                             0);
    this->check_tail_matches(this->draw_on_host(probe_count, eng_sampled),
                             this->draw_on_host(probe_count, eng_drawn),
                             0);
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "partial fisher yates draws distinct indices",
                     "[rng]",
                     rng_types_default_engine) {
    SKIP_IF(this->get_policy().is_cpu());

    const std::int64_t top = GENERATE_COPY(16, 1024);
    // `count == top` is a valid draw: it asks for every index of the population.
    const std::int64_t count = GENERATE_COPY(1, top / 2, top);

    auto& q = this->get_queue();
    auto engine_ = this->get_device_engine(default_seed);
    auto sample = this->allocate_indices(count);
    partial_fisher_yates_shuffle<std::int32_t>(q, sample, top, engine_).wait_and_throw();

    std::vector<bool> seen(top, false);
    for (std::int64_t i = 0; i < count; ++i) {
        const std::int32_t value = sample.get_data()[i];
        REQUIRE(value >= 0);
        REQUIRE(value < top);
        REQUIRE(!seen[value]);
        seen[value] = true;
    }
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "host partial fisher yates draws distinct indices",
                     "[rng]",
                     rng_types_default_engine) {
    const std::int64_t top = GENERATE_COPY(16, 1024);
    const std::int64_t count = GENERATE_COPY(1, top / 2, top);

    auto sample = ndarray<std::int32_t, 1>::empty({ count });
    host_engine engine_(default_seed);
    partial_fisher_yates_shuffle<std::int32_t>(sample, top, engine_);

    std::vector<bool> seen(top, false);
    for (std::int64_t i = 0; i < count; ++i) {
        const std::int32_t value = sample.get_data()[i];
        REQUIRE(value >= 0);
        REQUIRE(value < top);
        REQUIRE(!seen[value]);
        seen[value] = true;
    }
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "partial fisher yates composes on one engine",
                     "[rng]",
                     rng_types_default_engine) {
    SKIP_IF(this->get_policy().is_cpu());

    constexpr std::int64_t count = 64;
    constexpr std::int64_t top = 4096;

    auto& q = this->get_queue();
    auto first = this->allocate_indices(count);
    auto second = this->allocate_indices(count);

    auto engine_ = this->get_device_engine(default_seed);
    partial_fisher_yates_shuffle<std::int32_t>(q, first, top, engine_).wait_and_throw();
    partial_fisher_yates_shuffle<std::int32_t>(q, second, top, engine_).wait_and_throw();
    REQUIRE(std::vector<std::int32_t>(first.get_data(), first.get_data() + count) !=
            std::vector<std::int32_t>(second.get_data(), second.get_data() + count));

    // The one-shot overload restarts the stream, so it has to repeat itself.
    partial_fisher_yates_shuffle<std::int32_t>(q, first, top, default_seed).wait_and_throw();
    partial_fisher_yates_shuffle<std::int32_t>(q, second, top, default_seed).wait_and_throw();
    REQUIRE(std::vector<std::int32_t>(first.get_data(), first.get_data() + count) ==
            std::vector<std::int32_t>(second.get_data(), second.get_data() + count));
}

TEMPLATE_LIST_TEST_M(rng_test,
                     "host partial fisher yates composes on one engine",
                     "[rng]",
                     rng_types_default_engine) {
    constexpr std::int64_t count = 64;
    constexpr std::int64_t top = 4096;

    auto first = ndarray<std::int32_t, 1>::empty({ count });
    auto second = ndarray<std::int32_t, 1>::empty({ count });

    host_engine engine_(default_seed);
    partial_fisher_yates_shuffle<std::int32_t>(first, top, engine_);
    partial_fisher_yates_shuffle<std::int32_t>(second, top, engine_);
    REQUIRE(std::vector<std::int32_t>(first.get_data(), first.get_data() + count) !=
            std::vector<std::int32_t>(second.get_data(), second.get_data() + count));

    partial_fisher_yates_shuffle<std::int32_t>(first, top, default_seed);
    partial_fisher_yates_shuffle<std::int32_t>(second, top, default_seed);
    REQUIRE(std::vector<std::int32_t>(first.get_data(), first.get_data() + count) ==
            std::vector<std::int32_t>(second.get_data(), second.get_data() + count));
}

//TODO: add engine collection test + separate host_engine tests

} // namespace oneapi::dal::backend::primitives::test
