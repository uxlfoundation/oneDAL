#include "oneapi/dal/backend/primitives/frontier.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::backend::primitives::test {

TEST("frontier can be created", "[frontier]") {
    sycl::queue queue{ sycl::default_selector_v };

    const std::size_t num_items = 100;
    REQUIRE(num_items > 0);
    // auto f = frontier<std::uint32_t>(queue, num_items);

    // auto view = f.get_device_view();
    // view.insert(0);

} // TEST

} // namespace oneapi::dal::backend::primitives::test
