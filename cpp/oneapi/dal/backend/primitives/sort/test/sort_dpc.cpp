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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;

template <typename TestType>
class sort_with_indices_test : public te::policy_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Index = std::tuple_element_t<1, TestType>;

    auto allocate_arrays(Index elem_count) {
        auto& q = this->get_queue();
        auto val = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto ind = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        Index* ind_ptr = ind.get_mutable_data();
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(elem_count), [=](sycl::item<1> item) {
                 Index ind = item.get_id()[0];
                 ind_ptr[ind] = ind;
             });
         }).wait_and_throw();

        return std::make_tuple(val, ind);
    }

    void fill_uniform(ndarray<Float, 1>& val, Float a, Float b, std::int64_t seed = 777) {
        Index elem_count = de::integral_cast<Index>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> distr(a, b);

        // move generation to device when rng is available there
        auto val_host = ndarray<Float, 1>::empty({ val.get_count() });
        Float* val_ptr = val_host.get_mutable_data();
        for (Index el = 0; el < elem_count; el++) {
            val_ptr[el] = distr(rng);
        }

        auto& q = this->get_queue();
        val.assign(q, val_host.to_device(q)).wait_and_throw();
    }

    auto create_reference_on_host(const ndarray<Float, 1>& val) {
        return val.to_host(this->get_queue());
    }

    void check_sort(ndarray<Float, 1>& val, ndarray<Index, 1>& ind) {
        INFO("create reference");
        auto ref = create_reference_on_host(val);

        INFO("run sort with indices");
        auto event = radix_sort_indices_inplace<Float, Index>{ this->get_queue() }(val, ind);
        event.wait_and_throw();

        check_results(val, ind, ref);
    }

    void check_results(const ndarray<Float, 1>& val,
                       const ndarray<Index, 1> ind,
                       const ndarray<Float, 1>& ref) {
        const Float* ref_ptr = ref.get_data();

        const auto val_host = val.to_host(this->get_queue());
        const Float* val_ptr = val_host.get_data();

        const auto ind_host = ind.to_host(this->get_queue());
        const Index* ind_ptr = ind_host.get_data();

        for (Index el = 0; el < val.get_count(); el++) {
            if (el < val.get_count() - 1 && val_ptr[el] > val_ptr[el + 1]) {
                CAPTURE(el, val_ptr[el], el + 1, val_ptr[el + 1]);
                FAIL("elements are placed in inapropriate order");
            }

            if (val_ptr[el] != ref_ptr[ind_ptr[el]]) {
                CAPTURE(el, val_ptr[el], ind_ptr[el], ref_ptr[ind_ptr[el]]);
                FAIL("result elements indices are incorrect");
            }
        }
    }
};

template <typename Integer>
class sort_test : public te::policy_fixture {
public:
    auto allocate_arrays(std::int64_t vector_count, std::int64_t elem_count) {
        auto& q = this->get_queue();
        auto val =
            ndarray<Integer, 2>::empty(q, { vector_count, elem_count }, sycl::usm::alloc::device);

        return val;
    }

    void fill_uniform(ndarray<Integer, 2>& val,
                      std::int64_t a,
                      std::int64_t b,
                      std::int64_t seed = 777) {
        std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val.get_dimension(0));
        std::uint32_t elem_count = de::integral_cast<std::uint32_t>(val.get_dimension(1));

        std::mt19937 rng(seed);
        std::uniform_int_distribution<Integer> distr(a, b);

        // move generation to device when rng is available there
        auto val_host = ndarray<Integer, 2>::empty({ val.get_shape() });
        Integer* val_ptr = val_host.get_mutable_data();
        for (std::uint32_t vec = 0; vec < vector_count; vec++) {
            for (std::uint32_t el = 0; el < elem_count; el++) {
                val_ptr[vec * elem_count + el] = distr(rng);
            }
        }

        auto& q = this->get_queue();
        val.assign(q, val_host.to_device(q)).wait_and_throw();
    }

    void check_sort(ndarray<Integer, 2>& val, std::int64_t sorted_elem_count) {
        auto& q = this->get_queue();
        std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val.get_dimension(0));
        std::uint32_t elem_count = de::integral_cast<std::uint32_t>(val.get_dimension(1));

        INFO("create reference");
        auto ref = create_reference_on_host(val, sorted_elem_count);

        INFO("allocate auxiliary buffers");
        auto val_out = ndarray<Integer, 2>::empty(q, { vector_count, elem_count });

        INFO("run sort");
        radix_sort<Integer>{ this->get_queue() }(val, val_out, sorted_elem_count).wait_and_throw();

        check_results(val_out, ref, sorted_elem_count);
    }

    void check_results(const ndarray<Integer, 2>& val,
                       const ndarray<Integer, 2>& ref,
                       std::int64_t sorted_elem_count) {
        std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val.get_dimension(0));
        std::uint32_t elem_count = de::integral_cast<std::uint32_t>(val.get_dimension(1));
        auto val_host = val.to_host(this->get_queue());
        const Integer* val_ptr = val_host.get_data();
        const Integer* ref_ptr = ref.get_data();

        for (std::uint32_t vec = 0; vec < vector_count; vec++) {
            for (std::uint32_t el = 0; el < sorted_elem_count; el++) {
                if (val_ptr[vec * elem_count + el] != ref_ptr[vec * elem_count + el]) {
                    CAPTURE(elem_count,
                            sorted_elem_count,
                            vec,
                            el,
                            val_ptr[vec * elem_count + el],
                            ref_ptr[vec * elem_count + el]);
                    FAIL("result doesn't match reference");
                }
            }
        }
    }

    auto create_reference_on_host(const ndarray<Integer, 2>& in, std::int64_t sorted_elem_count) {
        auto ref = in.to_host(this->get_queue());
        auto ref_ptr = ref.get_mutable_data();

        for (std::int64_t vec = 0; vec < in.get_dimension(0); vec++) {
            std::sort(ref_ptr + vec * in.get_dimension(1),
                      ref_ptr + vec * in.get_dimension(1) + sorted_elem_count);
        }

        return ref;
    }
};

using sort_indices_types = COMBINE_TYPES((float, double), (std::int32_t, std::uint32_t));

TEMPLATE_LIST_TEST_M(sort_with_indices_test,
                     "basic sort with indices",
                     "[sort]",
                     sort_indices_types) {
    SKIP_IF(this->get_policy().is_cpu());

    std::int64_t elem_count = GENERATE_COPY(2, 10000);

    auto [val, ind] = this->allocate_arrays(elem_count);
    this->fill_uniform(val, -25., 25.);

    this->check_sort(val, ind);
}

using sort_indices_types_int = COMBINE_TYPES((std::int32_t, std::uint32_t));

TEMPLATE_LIST_TEST_M(sort_test, "basic sort", "[sort]", sort_indices_types_int) {
    SKIP_IF(this->get_policy().is_cpu());

    std::int64_t vector_count = GENERATE_COPY(1, 128);
    std::int64_t elem_count = GENERATE_COPY(2, 55, 1024);
    std::int64_t sorted_elem_count = elem_count - (elem_count > 2 ? GENERATE_COPY(0, 12) : 0);

    auto val = this->allocate_arrays(vector_count, elem_count);
    this->fill_uniform(val, 0, 50);
    this->check_sort(val, sorted_elem_count);
}

} // namespace oneapi::dal::backend::primitives::test
