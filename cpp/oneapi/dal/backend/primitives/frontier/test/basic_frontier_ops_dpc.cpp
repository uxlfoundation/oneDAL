#include "oneapi/dal/backend/primitives/frontier.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::backend::primitives::test {

void print_device_name(sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto device_name = device.get_info<sycl::info::device::name>();
    std::cout << "Running on device: " << device_name << std::endl;
}

template<typename T>
void print_frontier(const T* data, size_t num_items) {
    size_t element_bitsize = sizeof(T) * 8;
    for (size_t i = 0; i < num_items; ++i) {
        size_t offset = num_items / element_bitsize;
        size_t bit = i % element_bitsize;
        std::cout << ((data[offset] & (static_cast<T>(1) << bit)) ? "1" : "0");
    }       
}

TEST("frontier basic operations", "[frontier]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();
    print_device_name(queue);

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
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();
    print_device_name(queue);

    const std::size_t num_items = 100;
    auto f = frontier<std::uint32_t, sycl::usm::alloc::device>(queue, num_items);

    SECTION("empty function", "[frontier]") {
        REQUIRE(f.empty() == true);
        queue.single_task([=, view = f.get_device_view()]() {
            view.insert(0);
        }).wait_and_throw();
        std::cout << (f.check(0) ? "Frontier is not empty" : "Frontier is empty") << std::endl;
        std::vector<std::uint32_t> data_layer_cpy(f.get_data_size());
        REQUIRE(f.empty() == false);
    }

    SECTION("insert function", "[frontier]") {
        f.insert(0);
        f.insert(2);
        f.insert(bitset<std::uint32_t>::element_bitsize);

        auto* data_layer = f.get_data_ptr();
        auto* mlb_layer = f.get_mlb_ptr();

        bool* check = sycl::malloc_device<bool>(1, queue);
        queue.single_task([=, view = f.get_device_view()]() {
            check[0] = data_layer[0] == static_cast<uint32_t>(5);
            check[0] &= data_layer[1] == static_cast<uint32_t>(1);
            check[0] &= mlb_layer[0] == static_cast<uint32_t>(3);
        }).wait_and_throw();
        bool tmp_check;
        queue.copy(check, &tmp_check, 1).wait_and_throw();
        REQUIRE(tmp_check == true);
        sycl::free(check, queue);
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
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();

    const std::size_t num_items = 100000;
    auto f = frontier<std::uint32_t, sycl::usm::alloc::host>(queue, num_items);

    queue.single_task([=, view = f.get_device_view()]() {
        view.insert(0);
        view.insert(2);
        view.insert(32);
        view.insert(33);
        view.insert(64);
        view.insert(65);
        view.insert(95);
    }).wait_and_throw();

    auto e = f.compute_active_frontier();
    e.wait();

    auto offsets_size = f.get_offsets_size_ptr();

    REQUIRE(offsets_size[0] == 3);
    f.clear();
} // TEST "compute active frontier"

} // namespace oneapi::dal::backend::primitives::test
