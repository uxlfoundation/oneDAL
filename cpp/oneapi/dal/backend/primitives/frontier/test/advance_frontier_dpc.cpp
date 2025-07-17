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
void print_frontier(const T* data, size_t count, size_t num_items) {
    size_t element_bitsize = sizeof(T) * 8;
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < element_bitsize; ++j) {
            std::cout << ((data[i] & (static_cast<T>(1) << j)) ? "1" : "0");
        }
    }
}

TEST("test advance operation", "[advance]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();
    print_device_name(queue);

    std::vector<std::uint32_t> row_ptr = {0, 2, 4, 6};
    std::vector<std::uint32_t> col_indices = {1, 2, 0, 2, 0, 1};
    std::vector<std::uint32_t> weights = {1, 1, 1, 1, 1, 1};

    auto graph = csr_graph(queue, row_ptr, col_indices, weights);
    auto first_frontier = frontier<std::uint32_t>(queue, graph.get_vertex_count(), sycl::usm::alloc::device);
    auto second_frontier = frontier<std::uint32_t>(queue, graph.get_vertex_count(), sycl::usm::alloc::device);
    auto third_frontier = frontier<std::uint32_t>(queue, graph.get_vertex_count(), sycl::usm::alloc::device);

    first_frontier.insert(0);

    REQUIRE(first_frontier.check(0) == true);
    REQUIRE(first_frontier.check(1) == false);
    REQUIRE(first_frontier.check(2) == false);
    
    std::cout << "First Frontier: ";
    for (size_t i = 0; i < graph.get_vertex_count(); ++i) {
        std::cout << "Vertex " << i << ": " << (first_frontier.check(i) ? "in frontier" : "not in frontier") << std::endl;
    }
    
    advance(graph, first_frontier, second_frontier, [=](auto vertex, auto neighbor, auto edge, auto weight) {
        return true; // Always advance
    }).wait_and_throw();

    REQUIRE(second_frontier.check(0) == false);
    REQUIRE(second_frontier.check(1) == true);
    REQUIRE(second_frontier.check(2) == true);
    
    std::cout << "Second Frontier: ";
    for (size_t i = 0; i < graph.get_vertex_count(); ++i) {
        std::cout << "Vertex " << i << ": " << (second_frontier.check(i) ? "in frontier" : "not in frontier") << std::endl;
    }

    advance(graph, second_frontier, third_frontier, [=](auto vertex, auto neighbor, auto edge, auto weight) {
        return true; // Always advance
    }).wait_and_throw();

    std::cout << "Third Frontier: ";
    for (size_t i = 0; i < graph.get_vertex_count(); ++i) {
        std::cout << "Vertex " << i << ": " << (third_frontier.check(i) ? "in frontier" : "not in frontier") << std::endl;
    }

    REQUIRE(third_frontier.check(0) == true);
    REQUIRE(third_frontier.check(1) == true);
    REQUIRE(third_frontier.check(2) == true);

} // TEST "test advance operation"

} // namespace oneapi::dal::backend::primitives::test
