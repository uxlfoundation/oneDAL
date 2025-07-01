#include "oneapi/dal/backend/primitives/frontier.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::backend::primitives::test {

template <typename Param>
class frontier_test
        : public oneapi::dal::test::engine::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;

    void generate_input(std::int64_t n = -1, std::int64_t p = -1) {
        if (n == -1 || p == -1) {
            this->n_ = GENERATE(7, 827, 13, 216);
            this->p_ = GENERATE(4, 17, 41, 256);
        }
        else {
            this->n_ = n;
            this->p_ = p;
        }
        const auto dataframe = GENERATE_DATAFRAME(
            oneapi::dal::test::engine::dataframe_builder{ n_, p_ }.fill_uniform(-0.5, 0.5));
        this->data_ = dataframe.get_table(this->get_homogen_table_id());
    }

    void measure_time() {
        // Implement the performance measurement logic here
    }
};

TEMPLATE_LIST_TEST_M(frontier_test,
                     "frontier test",
                     "[frontier]",
                     oneapi::dal::test::engine::float_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate_input();
    this->measure_time();
}

} // namespace oneapi::dal::backend::primitives::test
