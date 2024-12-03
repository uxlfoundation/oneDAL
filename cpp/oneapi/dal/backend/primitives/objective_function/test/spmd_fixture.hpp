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

#include "fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace de = dal::detail;

template <typename Param>
class logloss_spmd_test : public logloss_test<Param> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    bool fit_intercept_ = std::tuple_element_t<1, Param>::value;
    // using float_t = Param;
    using comm_t = te::thread_communicator<spmd::device_memory_access::usm>;

    std::vector<std::pair<table, ndview<std::int32_t, 1>>>
    split_input_data(const table& data, ndview<std::int32_t, 1> labels, std::int64_t split_count) {
        ONEDAL_ASSERT(split_count > 0);
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();
        const std::int64_t block_size_regular = row_count / split_count;
        const std::int64_t block_size_tail = row_count % split_count;

        std::vector<std::pair<table, ndview<std::int32_t, 1>>> result(split_count);

        std::int64_t row_offset = 0;
        for (std::int64_t i = 0; i < split_count; i++) {
            const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
            const std::int64_t block_size = block_size_regular + tail;
            REQUIRE(0l < block_size);
            const auto row_range = range{ row_offset, row_offset + block_size };
            const auto block = te::get_table_block<float_t>(this->get_policy(), data, row_range);
            const auto labels_range = labels.get_slice(row_offset, row_offset + block_size);
            result[i].first = homogen_table::wrap(block, block_size, column_count);
            result[i].second = labels_range;
            row_offset += block_size;
        }
        return result;
    }

    std::vector<logloss_function<float_t>> get_functors(comm_t& comm,
                                                        std::int64_t thr_cnt,
                                                        table data,
                                                        ndview<std::int32_t, 1>& labels,
                                                        double L2,
                                                        bool fit_intercept) {
        auto input = split_input_data(data, labels, thr_cnt);

        std::vector<logloss_function<float_t>> funcs;
        funcs.reserve(thr_cnt);

        for (int i = 0; i < thr_cnt; ++i) {
            auto functor = logloss_function<float_t>(this->get_queue(),
                                                     comm,
                                                     input[i].first,
                                                     input[i].second,
                                                     float_t(L2),
                                                     fit_intercept);
            funcs.push_back(functor);
        }
        return funcs;
    }

    void run_spmd(std::int64_t thr_cnt, double L2, bool fit_intercept) {
        if (thr_cnt == -1) {
            thr_cnt = GENERATE(2, 4, 5);
        }

        constexpr float_t rtol = sizeof(float_t) > 4 ? 1e-6 : 1e-4;
        constexpr float_t atol = sizeof(float_t) > 4 ? 1e-6 : 1e-2;
        auto labels_gpu = this->labels_.to_device(this->get_queue());

        std::int64_t dim = this->p_ + (std::int64_t)(fit_intercept);
        auto param_array = row_accessor<const float_t>{ this->params_ }.pull(this->get_queue());
        auto params_host = ndarray<float_t, 1>::wrap(param_array.get_data(), { dim });
        auto params_gpu = params_host.to_device(this->get_queue());

        comm_t comm{ this->get_queue(), thr_cnt };

        // logloss_function has different regularization so we need to multiply it by 2 to allign with other implementations
        auto funcs = get_functors(comm, thr_cnt, this->data_, labels_gpu, L2 * 2, fit_intercept);
        std::int64_t num_checks = 5;

        std::vector<ndarray<float_t, 1>> vecs_host(num_checks), vecs_gpu(num_checks);
        rng<float_t> rn_gen;
        for (std::int64_t ij = 0; ij < num_checks; ++ij) {
            daal_engine eng(2007 + dim * num_checks + ij);
            vecs_host[ij] =
                (ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host));
            rn_gen.uniform(dim, vecs_host[ij].get_mutable_data(), eng, -1.0, 1.0);
            vecs_gpu[ij] = vecs_host[ij].to_device(this->get_queue());
        }

        const auto results = comm.map([&](std::int64_t rank) {
            sycl::event::wait_and_throw(funcs[rank].update_x(params_gpu, true, {}));
            base_matrix_operator<float_t>& hessp = funcs[rank].get_hessian_product();
            std::vector<ndarray<float_t, 1>> out_vecs(num_checks);
            for (std::int64_t ij = 0; ij < num_checks; ++ij) {
                out_vecs[ij] = ndarray<float_t, 1>::empty(this->get_queue(),
                                                          { dim },
                                                          sycl::usm::alloc::device);
                hessp(vecs_gpu[ij], out_vecs[ij], {}).wait_and_throw();
            }
            auto res =
                std::make_tuple(funcs[rank].get_value(), funcs[rank].get_gradient(), out_vecs);
            return res;
        });

        REQUIRE(results.size() == dal::detail::integral_cast<std::size_t>(thr_cnt));

        auto data_array = row_accessor<const float_t>{ this->data_ }.pull(this->get_queue());
        auto data_host = ndarray<float_t, 2>::wrap(data_array.get_data(), { this->n_, this->p_ });
        auto probs_gth =
            ndarray<float_t, 1>::empty(this->get_queue(), { this->n_ }, sycl::usm::alloc::host);
        auto grad_gth =
            ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);
        auto hess_gth = ndarray<float_t, 2>::empty(this->get_queue(),
                                                   { this->p_ + 1, this->p_ + 1 },
                                                   sycl::usm::alloc::host);
        auto hessp_gth =
            ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);

        this->naive_probabilities(data_host, params_host, this->labels_, probs_gth, fit_intercept);

        float_t logloss_gth = this->naive_logloss(data_host,
                                                  params_host,
                                                  this->labels_,
                                                  float_t(0.0),
                                                  float_t(L2),
                                                  fit_intercept);
        this->naive_derivative(data_host,
                               probs_gth,
                               params_host,
                               this->labels_,
                               grad_gth,
                               float_t(0.0),
                               float_t(L2),
                               fit_intercept);
        for (std::int64_t k = 0; k < thr_cnt; ++k) {
            IS_CLOSE(float_t, std::get<0>(results[k]), logloss_gth, rtol, atol);
        }

        for (int k = 0; k < thr_cnt; ++k) {
            auto grad_host = std::get<1>(results[k]).to_host(this->get_queue());
            for (int j = 0; j < dim; ++j) {
                IS_CLOSE(float_t, grad_host.at(j), grad_gth.at(j), rtol, atol);
            }
        }

        this->naive_hessian(data_host, probs_gth, hess_gth, float_t(L2), fit_intercept);

        const std::int64_t st = fit_intercept ? 0 : 1;

        for (std::int64_t ij = 0; ij < num_checks; ++ij) {
            for (std::int64_t i = st; i < this->p_ + 1; ++i) {
                float_t correct = 0;
                for (std::int64_t j = st; j < this->p_ + 1; ++j) {
                    correct += vecs_host[ij].at(j - st) * hess_gth.at(i, j);
                }
                hessp_gth.at(i - st) = correct;
            }

            for (std::int64_t k = 0; k < thr_cnt; ++k) {
                auto hessp_host = std::get<2>(results[k])[ij].to_host(this->get_queue());
                for (std::int64_t j = 0; j < dim; ++j) {
                    IS_CLOSE(float_t, hessp_host.at(j), hessp_gth.at(j), rtol, atol);
                }
            }
        }
    }
};

} // namespace oneapi::dal::backend::primitives::test
