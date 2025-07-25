#include "oneapi/dal/backend/primitives/frontier/frontier.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace pr = dal::backend::primitives;

void print_device_name(sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto device_name = device.get_info<sycl::info::device::name>();
    std::cout << "Running on device: " << device_name << std::endl;
}

template<typename T>
void print_frontier(const T* data, size_t count, size_t num_items) {
    size_t element_bitsize = sizeof(T) * 8;
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < element_bitsize; ++j) {
            std::cout << ((data[i] & (static_cast<T>(1) << j)) ? "1" : "0");
        }
    }
}

TEST("frontier queue basic operations", "[frontier]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();
    print_device_name(queue);

    const std::size_t num_items = 100;
    auto f = pr::frontier<std::uint32_t>(queue, num_items, sycl::usm::alloc::shared);

    REQUIRE(f.empty() == true);
    f.insert(0);
    f.insert(2);
    f.insert(32);
    f.insert(128);

    REQUIRE(f.empty() == false);

    REQUIRE(f.get_data().at_device(queue, 0) == 5);
    REQUIRE(f.get_data().at_device(queue, 1) == 1);
    REQUIRE(f.get_mlb().at_device(queue, 0) == 19);

    REQUIRE(f.check(0) == true);
    REQUIRE(f.check(1) == false);
    REQUIRE(f.check(2) == true);
    REQUIRE(f.check(3) == false);
    REQUIRE(f.check(bitset<std::uint32_t>::element_bitsize) == true);
    REQUIRE(f.check(bitset<std::uint32_t>::element_bitsize + 1) == false);
    
    auto e = f.compute_active_frontier();
    REQUIRE(f.get_offsets_size().at_device(queue, 0, {e}) == 3);
    for (int i = 0; i < 3; i++) {
        std::cout << "Active frontier item " << i << ": " << f.get_offsets().at_device(queue, i) << std::endl;
    }

    f.clear();
    REQUIRE(f.empty() == true);

} // TEST "frontier queue operations"

} // namespace oneapi::dal::backend::primitives::test
