#include "oneapi/dal/backend/primitives/frontier.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::backend::primitives::test {

TEST("frontier basic operations", "[frontier]") {
    sycl::queue queue{ sycl::default_selector_v };

    const std::size_t num_items = 100;
    auto f = frontier<std::uint32_t>(queue, num_items);
    auto view = f.get_device_view();
    size_t expected_data_layer_size = (num_items + bitset<std::uint32_t>::element_bitsize - 1) /
                                      bitset<std::uint32_t>::element_bitsize;
    size_t expected_mlb_layer_size =
        (expected_data_layer_size + bitset<std::uint32_t>::element_bitsize - 1) /
        bitset<std::uint32_t>::element_bitsize;

    SECTION("frontier has correct data layer size", "[frontier]") {
        REQUIRE(f.get_data_size() == expected_data_layer_size);

        REQUIRE(f.get_mlb_size() == expected_mlb_layer_size);
    }

    SECTION("frontier can insert elements", "[frontier]") {
        view.insert(0);
        view.insert(2);
        view.insert(bitset<std::uint32_t>::element_bitsize);

        auto data_layer = f.get_data_ptr();
        auto mlb_layer = f.get_mlb_ptr();

        REQUIRE(data_layer[0] == static_cast<uint32_t>(5));
        REQUIRE(data_layer[1] == static_cast<uint32_t>(1));
        REQUIRE(mlb_layer[0] == static_cast<uint32_t>(3));
    }

    SECTION("frontier can check elements", "[frontier]") {
        view.insert(0);
        view.insert(2);
        view.insert(32);
        view.insert(64);

        REQUIRE(view.check(0) == true);
        REQUIRE(view.check(1) == false);
        REQUIRE(view.check(2) == true);
        REQUIRE(view.check(3) == false);
        REQUIRE(view.check(31) == false);
        REQUIRE(view.check(32) == true);
        REQUIRE(view.check(33) == false);
        REQUIRE(view.check(63) == false);
        REQUIRE(view.check(64) == true);
        REQUIRE(view.check(65) == false);
    }

} // TEST "frontier basic operations"

TEST("frontier queue basic operations", "[frontier]") {
    sycl::queue queue{ sycl::default_selector_v };

    const std::size_t num_items = 100;
    auto f = frontier<std::uint32_t>(queue, num_items);

    SECTION("empty function", "[frontier]") {
        REQUIRE(f.empty() == true);
        auto view = f.get_device_view();
        view.insert(0);
        REQUIRE(f.empty() == false);
    }

    SECTION("insert function", "[frontier]") {
        f.insert(0);
        f.insert(2);
        f.insert(bitset<std::uint32_t>::element_bitsize);

        auto data_layer = f.get_data_ptr();
        auto mlb_layer = f.get_mlb_ptr();

        REQUIRE(data_layer[0] == static_cast<uint32_t>(5));
        REQUIRE(data_layer[1] == static_cast<uint32_t>(1));
        REQUIRE(mlb_layer[0] == static_cast<uint32_t>(3));
    }

    SECTION("check function", "[frontier]") {
        f.insert(0);
        REQUIRE(f.check(0) == true);
        REQUIRE(f.check(1) == false);
    }

    SECTION("clear function", "[frontier]") {
        f.insert(0);
        f.insert(2);
        f.insert(bitset<std::uint32_t>::element_bitsize);
        REQUIRE(f.empty() == false);
        f.clear();
        REQUIRE(f.empty() == true);
    }

} // TEST "frontier queue operations"

TEST("compute active frontier", "[frontier]") {
    sycl::queue queue{ sycl::default_selector_v };

    const std::size_t num_items = 100000;
    auto f = frontier<std::uint32_t>(queue, num_items);
    auto view = f.get_device_view();

    view.insert(0);
    view.insert(2);
    view.insert(32);
    view.insert(1024);

    auto e = f.compute_active_frontier();
    e.wait();

    // auto offsets = f.get_offsets_ptr();
    auto offsets_size = f.get_offsets_size_ptr();

    REQUIRE(offsets_size[0] == 2);
} // TEST "compute active frontier"

} // namespace oneapi::dal::backend::primitives::test
