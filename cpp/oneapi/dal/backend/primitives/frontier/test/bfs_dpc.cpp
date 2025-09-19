#include "oneapi/dal/backend/primitives/frontier.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include <queue>
#include <vector>
#include <map>
#include <chrono>

namespace oneapi::dal::backend::primitives::test {

namespace pr = dal::backend::primitives;

void print_device_name(sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto device_name = device.get_info<sycl::info::device::name>();
    std::cout << "Running on device: " << device_name << std::endl;
}

template <typename T>
std::vector<std::uint32_t> host_bfs(std::vector<T>& row_offsets,
                                    std::vector<T>& col_indices,
                                    T src) {
    std::vector<std::uint32_t> distances(row_offsets.size() - 1,
                                         std::numeric_limits<std::uint32_t>::max());
    std::queue<T> q;
    q.push(src);
    distances[src] = 0;

    while (!q.empty()) {
        T node = q.front();
        q.pop();

        for (size_t i = row_offsets[node]; i < row_offsets[node + 1]; ++i) {
            auto neighbor = col_indices[i];
            if (distances[neighbor] == std::numeric_limits<std::uint32_t>::max()) {
                distances[neighbor] = distances[node] + 1;
                q.push(neighbor);
            }
        }
    }
    return distances;
}

TEST("test advance operation", "[advance]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();
    print_device_name(queue);

    std::vector<std::uint32_t> row_ptr = {
        0,   2,   5,   9,   13,  17,  21,  25,  29,  33,  37,  41,  45,  49,  53,  57,  61,
        65,  69,  73,  77,  81,  85,  89,  93,  97,  101, 105, 109, 113, 117, 121, 125, 129,
        133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197,
        201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253, 257, 261, 265,
        269, 273, 277, 281, 285, 289, 293, 297, 301, 305, 309, 313, 317, 321, 325, 329, 333,
        337, 341, 345, 349, 353, 357, 361, 365, 369, 373, 377, 381, 385, 389, 393, 397, 401,
        405, 409, 413, 417, 421, 425, 429, 433, 437, 441, 445, 449, 453, 457, 461, 465, 469,
        473, 477, 481, 485, 489, 493, 497, 501, 504, 506
    };
    std::vector<std::uint32_t> col_indices = {
        1,   2,   0,   2,   3,   0,   1,   3,   4,   1,   2,   4,   5,   2,   3,   5,   6,   3,
        4,   6,   7,   4,   5,   7,   8,   5,   6,   8,   9,   6,   7,   9,   10,  7,   8,   10,
        11,  8,   9,   11,  12,  9,   10,  12,  13,  10,  11,  13,  14,  11,  12,  14,  15,  12,
        13,  15,  16,  13,  14,  16,  17,  14,  15,  17,  18,  15,  16,  18,  19,  16,  17,  19,
        20,  17,  18,  20,  21,  18,  19,  21,  22,  19,  20,  22,  23,  20,  21,  23,  24,  21,
        22,  24,  25,  22,  23,  25,  26,  23,  24,  26,  27,  24,  25,  27,  28,  25,  26,  28,
        29,  26,  27,  29,  30,  27,  28,  30,  31,  28,  29,  31,  32,  29,  30,  32,  33,  30,
        31,  33,  34,  31,  32,  34,  35,  32,  33,  35,  36,  33,  34,  36,  37,  34,  35,  37,
        38,  35,  36,  38,  39,  36,  37,  39,  40,  37,  38,  40,  41,  38,  39,  41,  42,  39,
        40,  42,  43,  40,  41,  43,  44,  41,  42,  44,  45,  42,  43,  45,  46,  43,  44,  46,
        47,  44,  45,  47,  48,  45,  46,  48,  49,  46,  47,  49,  50,  47,  48,  50,  51,  48,
        49,  51,  52,  49,  50,  52,  53,  50,  51,  53,  54,  51,  52,  54,  55,  52,  53,  55,
        56,  53,  54,  56,  57,  54,  55,  57,  58,  55,  56,  58,  59,  56,  57,  59,  60,  57,
        58,  60,  61,  58,  59,  61,  62,  59,  60,  62,  63,  60,  61,  63,  64,  61,  62,  64,
        65,  62,  63,  65,  66,  63,  64,  66,  67,  64,  65,  67,  68,  65,  66,  68,  69,  66,
        67,  69,  70,  67,  68,  70,  71,  68,  69,  71,  72,  69,  70,  72,  73,  70,  71,  73,
        74,  71,  72,  74,  75,  72,  73,  75,  76,  73,  74,  76,  77,  74,  75,  77,  78,  75,
        76,  78,  79,  76,  77,  79,  80,  77,  78,  80,  81,  78,  79,  81,  82,  79,  80,  82,
        83,  80,  81,  83,  84,  81,  82,  84,  85,  82,  83,  85,  86,  83,  84,  86,  87,  84,
        85,  87,  88,  85,  86,  88,  89,  86,  87,  89,  90,  87,  88,  90,  91,  88,  89,  91,
        92,  89,  90,  92,  93,  90,  91,  93,  94,  91,  92,  94,  95,  92,  93,  95,  96,  93,
        94,  96,  97,  94,  95,  97,  98,  95,  96,  98,  99,  96,  97,  99,  100, 97,  98,  100,
        101, 98,  99,  101, 102, 99,  100, 102, 103, 100, 101, 103, 104, 101, 102, 104, 105, 102,
        103, 105, 106, 103, 104, 106, 107, 104, 105, 107, 108, 105, 106, 108, 109, 106, 107, 109,
        110, 107, 108, 110, 111, 108, 109, 111, 112, 109, 110, 112, 113, 110, 111, 113, 114, 111,
        112, 114, 115, 112, 113, 115, 116, 113, 114, 116, 117, 114, 115, 117, 118, 115, 116, 118,
        119, 116, 117, 119, 120, 117, 118, 120, 121, 118, 119, 121, 122, 119, 120, 122, 123, 120,
        121, 123, 124, 121, 122, 124, 125, 122, 123, 125, 126, 123, 124, 126, 127, 124, 125, 127,
        125, 126
    };
    std::vector<std::uint32_t> weights(col_indices.size(), 1);

    auto graph = pr::csr_graph(queue, row_ptr, col_indices, weights);
    size_t num_nodes = row_ptr.size() - 1;
    auto in_frontier = pr::frontier<std::uint32_t>(queue, num_nodes, sycl::usm::alloc::device);
    auto out_frontier = pr::frontier<std::uint32_t>(queue, num_nodes, sycl::usm::alloc::device);
    pr::ndarray<std::uint32_t, 1> distance =
        pr::ndarray<std::uint32_t, 1>::empty(queue,
                                             { static_cast<std::int64_t>(num_nodes) },
                                             sycl::usm::alloc::device);
    auto distance_ptr = distance.get_mutable_data();

    const std::uint32_t src = 0;

    queue
        .submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(num_nodes), [=](sycl::id<1> idx) {
                distance_ptr[idx] = idx[0] == src ? 0 : num_nodes + 1;
            });
        })
        .wait_and_throw();

    in_frontier.insert(src);
    size_t iter = 0;

    /// Start BFS
    while (!in_frontier.empty()) {
        auto e = pr::advance(graph,
                             in_frontier,
                             out_frontier,
                             [=](auto vertex, auto neighbor, auto edge, auto weight) {
                                 bool visited = distance_ptr[neighbor] < num_nodes + 1;
                                 if (!visited) {
                                     distance_ptr[neighbor] = iter + 1;
                                 }
                                 return !visited;
                             });
        e.wait_and_throw();
        iter++;
        pr::swap_frontiers(in_frontier, out_frontier);
        out_frontier.clear();
    }
    /// End BFS

    auto expected_distances = host_bfs(row_ptr, col_indices, src);
    auto actual_distances = distance.to_host(queue).get_data();
    for (size_t i = 0; i < num_nodes; ++i) {
        REQUIRE(actual_distances[i] == expected_distances[i]);
    }
} // TEST "test advance operation"

} // namespace oneapi::dal::backend::primitives::test
