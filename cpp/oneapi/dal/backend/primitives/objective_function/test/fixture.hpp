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

#include <type_traits>

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/backend/primitives/objective_function/logloss_functors.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/csr_table_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/detail/debug.hpp"

#include "oneapi/dal/backend/primitives/rng/rng.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <bool fit_intercept>
struct fit_intercept_tag {
    static constexpr bool value = fit_intercept;
};

using use_fit_intercept = fit_intercept_tag<true>;
using no_fit_intercept = fit_intercept_tag<false>;

using logloss_types = COMBINE_TYPES((float, double), (use_fit_intercept, no_fit_intercept));

#define IS_CLOSE(ftype, real, expected, rtol, atol) \
    REQUIRE(abs(real - expected) < atol);           \
    REQUIRE(abs(real - expected) / std::max(std::abs(expected), (ftype)1.0) < rtol);

template <typename Param>
class logloss_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    bool fit_intercept_ = std::tuple_element_t<1, Param>::value;

    void generate_input(std::int64_t n = -1, std::int64_t p = -1) {
        if (n == -1 || p == -1) {
            this->n_ = GENERATE(7, 827, 13, 216);
            this->p_ = GENERATE(4, 17, 41, 256);
        }
        else {
            this->n_ = n;
            this->p_ = p;
        }
        const auto dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ n_, p_ }.fill_uniform(-0.5, 0.5));
        const auto parameters =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, p_ + 1 }.fill_uniform(-1, 1));
        this->data_ = dataframe.get_table(this->get_homogen_table_id());
        this->params_ = parameters.get_table(this->get_homogen_table_id());
        this->labels_ =
            ndarray<std::int32_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);

        std::srand(2007 + n_);
        auto* const ptr_lab = this->labels_.get_mutable_data();
        for (std::int64_t i = 0; i < n_; ++i) {
            ptr_lab[i] = std::rand() % 2;
        }
    }

    void generate_sparse_input(std::int64_t n = -1, std::int64_t p = -1) {
        if (n == -1 || p == -1) {
            this->n_ = GENERATE(7, 827, 13, 216);
            this->p_ = GENERATE(4, 17, 41, 256);
        }
        else {
            this->n_ = n;
            this->p_ = p;
        }

        auto builder = te::csr_table_builder<float_t>(n_, p_, 0.3, sparse_indexing::zero_based);
        this->data_ = builder.build_csr_table(this->get_policy());
        this->dense_data_ = builder.build_dense_table(this->get_policy());

        const auto parameters =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, p_ + 1 }.fill_uniform(-1, 1));
        this->params_ = parameters.get_table(this->get_homogen_table_id());
        this->labels_ =
            ndarray<std::int32_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);
        std::srand(2007 + n_);
        auto* const ptr_lab = this->labels_.get_mutable_data();
        for (std::int64_t i = 0; i < n_; ++i) {
            ptr_lab[i] = std::rand() % 2;
        }
    }

    void run_test(const float_t L1 = 0,
                  const float_t L2 = 0,
                  bool fit_intercept = true,
                  bool batch_test = false) {
        auto data_array = row_accessor<const float_t>{ this->data_ }.pull(this->get_queue());
        auto data_host = ndarray<float_t, 2>::wrap(data_array.get_data(), { n_, p_ });

        std::int64_t dim = fit_intercept ? this->p_ + 1 : this->p_;

        auto param_array = row_accessor<const float_t>{ this->params_ }.pull(this->get_queue());
        auto params_host = ndarray<float_t, 1>::wrap(param_array.get_data(), { dim });
        test_input(data_host, params_host, this->labels_, L1, L2, fit_intercept, batch_test);

        SUCCEED();
    }

    void run_sparse_test(const float_t L2 = 0, bool fit_intercept = true) {
        constexpr float_t rtol = sizeof(float_t) > 4 ? 1e-6 : 5e-4;
        constexpr float_t atol = sizeof(float_t) > 4 ? 1e-6 : 1e-1;

        REQUIRE(this->data_.get_kind() == csr_table::kind());

        auto data_array = row_accessor<const float_t>{ this->dense_data_ }.pull(this->get_queue());
        auto data_host = ndarray<float_t, 2>::wrap(data_array.get_data(), { n_, p_ });

        std::int64_t dim = fit_intercept ? this->p_ + 1 : this->p_;
        auto param_array = row_accessor<const float_t>{ this->params_ }.pull(this->get_queue());
        auto params_host = ndarray<float_t, 1>::wrap(param_array.get_data(), { dim });
        auto params_gpu = params_host.to_device(this->get_queue());
        auto labels_gpu = this->labels_.to_device(this->get_queue());

        float_t gth_logloss =
            naive_logloss(data_host, params_host, this->labels_, float_t(0), L2, fit_intercept);

        auto gth_probs =
            ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);
        naive_probabilities(data_host, params_host, this->labels_, gth_probs, fit_intercept);

        auto gth_gradient =
            ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);
        naive_derivative(data_host,
                         gth_probs,
                         params_host,
                         this->labels_,
                         gth_gradient,
                         float_t(0),
                         L2,
                         fit_intercept);

        auto gth_hessian = ndarray<float_t, 2>::empty(this->get_queue(),
                                                      { p_ + 1, p_ + 1 },
                                                      sycl::usm::alloc::host);
        naive_hessian(data_host, gth_probs, gth_hessian, L2, fit_intercept);

        test_functors(data_,
                      labels_gpu,
                      params_gpu,
                      gth_gradient,
                      gth_hessian,
                      gth_logloss,
                      L2,
                      fit_intercept,
                      false,
                      rtol,
                      atol);
    }

    void test_gold_input(bool fit_intercept = true) {
        constexpr std::int64_t n = 5;
        constexpr std::int64_t p = 3;
        constexpr float_t data[n * p] = { 0.83731708,  -0.70899924, -1.23362082, 0.23468538,
                                          -0.10549413, 1.12902673,  -0.61035703, -1.55617932,
                                          0.60419908,  0.30589827,  0.63919892,  -0.23380754,
                                          2.38196927,  1.64158111,  0.13677077 };
        constexpr std::int32_t labels[n] = { 0, 1, 1, 0, 1 };
        constexpr float_t L1 = 0;
        constexpr float_t L2 = 3.123;
        constexpr float_t cur_param[p + 1] = { -0.2, 0.1, -1, 0.4 };

        auto data_host = ndarray<float_t, 2>::wrap(data, { n, p });

        this->data_ = homogen_table::wrap(data_host.get_data(), n, p);

        auto labels_host = ndarray<std::int32_t, 1>::wrap(labels, n);

        ndarray<float_t, 1> params_host;
        if (fit_intercept) {
            params_host = ndarray<float_t, 1>::wrap(cur_param, p + 1);
        }
        else {
            params_host = ndarray<float_t, 1>::wrap(cur_param + 1, p);
        }

        test_input(data_host, params_host, labels_host, L1, L2, fit_intercept);

        SUCCEED();
    }

    void test_input(const ndarray<float_t, 2>& data_host,
                    const ndarray<float_t, 1>& params_host,
                    const ndarray<std::int32_t, 1>& labels_host,
                    const float_t L1,
                    const float_t L2,
                    bool fit_intercept,
                    bool batch_test = false) {
        constexpr float_t rtol = sizeof(float_t) > 4 ? 1e-6 : 1e-4;
        constexpr float_t atol = sizeof(float_t) > 4 ? 1e-6 : 1e-1;
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);
        const std::int64_t dim = params_host.get_dimension(0);

        auto data_gpu = data_host.to_device(this->get_queue());
        auto labels_gpu = labels_host.to_device(this->get_queue());
        auto params_gpu = params_host.to_device(this->get_queue());

        auto out_predictions =
            ndarray<float_t, 1>::empty(this->get_queue(), { n }, sycl::usm::alloc::device);

        auto p_event = compute_probabilities(this->get_queue(),
                                             params_gpu,
                                             data_gpu,
                                             out_predictions,
                                             fit_intercept,
                                             {});
        p_event.wait_and_throw();
        auto predictions_host = out_predictions.to_host(this->get_queue(), {});

        const float_t logloss = test_predictions_and_logloss(data_host,
                                                             params_host,
                                                             labels_host,
                                                             predictions_host,
                                                             L1,
                                                             L2,
                                                             fit_intercept,
                                                             rtol,
                                                             atol);

        auto [out_logloss, out_e] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { 1 }, sycl::usm::alloc::device);
        sycl::event logloss_event = compute_logloss(this->get_queue(),
                                                    labels_gpu,
                                                    out_predictions,
                                                    out_logloss,
                                                    fit_intercept,
                                                    { out_e });
        sycl::event logloss_reg_event = add_regularization_loss(this->get_queue(),
                                                                params_gpu,
                                                                out_logloss,
                                                                L1,
                                                                L2,
                                                                fit_intercept,
                                                                { logloss_event });
        logloss_reg_event.wait_and_throw();

        const float_t val_logloss1 = out_logloss.to_host(this->get_queue(), {}).at(0);
        IS_CLOSE(float_t, val_logloss1, logloss, rtol, atol);

        auto fill_event = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});
        auto [out_derivative, out_der_e] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { dim }, sycl::usm::alloc::device);
        auto logloss_event_der = compute_logloss_with_der(this->get_queue(),
                                                          data_gpu,
                                                          labels_gpu,
                                                          out_predictions,
                                                          out_logloss,
                                                          out_derivative,
                                                          fit_intercept,
                                                          { fill_event, out_der_e });
        auto regul_logloss_and_der_event = add_regularization_gradient_loss(this->get_queue(),
                                                                            params_gpu,
                                                                            out_logloss,
                                                                            out_derivative,
                                                                            L1,
                                                                            L2,
                                                                            fit_intercept,
                                                                            { logloss_event_der });
        regul_logloss_and_der_event.wait_and_throw();
        auto out_derivative_host = out_derivative.to_host(this->get_queue());

        const float_t val_logloss2 = out_logloss.to_host(this->get_queue(), {}).at(0);
        IS_CLOSE(float_t, val_logloss2, logloss, rtol, atol);

        auto [out_derivative2, out_der_e2] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { dim }, sycl::usm::alloc::device);
        auto der_event = compute_derivative(this->get_queue(),
                                            data_gpu,
                                            labels_gpu,
                                            out_predictions,
                                            out_derivative2,
                                            fit_intercept,
                                            { out_der_e2 });
        auto der_reg_event = add_regularization_gradient(this->get_queue(),
                                                         params_gpu,
                                                         out_derivative2,
                                                         L1,
                                                         L2,
                                                         fit_intercept,
                                                         { der_event });

        der_reg_event.wait_and_throw();
        auto out_derivative_host2 = out_derivative2.to_host(this->get_queue());

        for (std::int64_t i = 0; i < dim; ++i) {
            REQUIRE(abs(out_derivative_host.at(i) - out_derivative_host2.at(i)) < atol);
        }
        auto [out_hessian, out_hess_e] = ndarray<float_t, 2>::zeros(this->get_queue(),
                                                                    { p + 1, p + 1 },
                                                                    sycl::usm::alloc::device);
        auto hess_event = compute_hessian(this->get_queue(),
                                          data_gpu,
                                          labels_gpu,
                                          out_predictions,
                                          out_hessian,
                                          L1,
                                          L2,
                                          fit_intercept,
                                          { out_hess_e });

        auto hessian_host = out_hessian.to_host(this->get_queue(), { hess_event });

        test_formula_derivative(data_host,
                                predictions_host,
                                params_host,
                                labels_host,
                                out_derivative_host,
                                L1,
                                L2,
                                fit_intercept,
                                rtol,
                                atol);
        test_formula_hessian(data_host,
                             predictions_host,
                             hessian_host,
                             L2,
                             fit_intercept,
                             rtol,
                             atol);

        if (L1 == 0) {
            test_functors(data_,
                          labels_gpu,
                          params_gpu,
                          out_derivative_host,
                          hessian_host,
                          logloss,
                          L2,
                          fit_intercept,
                          batch_test,
                          rtol,
                          atol);
        }
    }

    float_t clip_prob(float_t prob) {
        constexpr float_t bottom = sizeof(float_t) > 4 ? 1e-15 : 1e-7;
        constexpr float_t top = float_t(1.0) - bottom;
        if (prob < bottom) {
            prob = bottom;
        }
        if (prob > top) {
            prob = top;
        }
        return prob;
    }

    float_t test_predictions_and_logloss(const ndview<float_t, 2>& data_host,
                                         const ndview<float_t, 1>& params_host,
                                         const ndview<std::int32_t, 1>& labels_host,
                                         const ndview<float_t, 1>& probabilities,
                                         const float_t L1,
                                         const float_t L2,
                                         bool fit_intercept,
                                         const float_t rtol = 1e-3,
                                         const float_t atol = 1e-3) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);
        const std::int64_t start_ind = fit_intercept ? 1 : 0;
        float_t logloss = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            float_t pred = 0;

            for (std::int64_t j = 0; j < p; ++j) {
                pred += params_host.at(j + start_ind) * data_host.at(i, j);
            }
            if (fit_intercept) {
                pred += params_host.at(0);
            }
            float_t prob = clip_prob(float_t(1.0) / (1 + std::exp(-pred)));
            logloss -=
                labels_host.at(i) * std::log(prob) + (1 - labels_host.at(i)) * std::log(1 - prob);
            float_t out_val = probabilities.at(i);
            REQUIRE(abs(out_val - prob) < atol);
        }
        for (std::int64_t i = 0; i < p; ++i) {
            float_t param = params_host.at(i + start_ind);
            logloss += L1 * abs(param);
            logloss += L2 * param * param;
        }
        return logloss;
    }

    float_t naive_logloss(const ndview<float_t, 2>& data_host,
                          const ndview<float_t, 1>& params_host,
                          const ndview<std::int32_t, 1>& labels_host,
                          float_t L1,
                          float_t L2,
                          bool fit_intercept) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);

        // We use double for gth computation to achieve better precision
        double logloss = 0;
        std::int64_t st = fit_intercept;
        for (std::int64_t i = 0; i < n; ++i) {
            double pred = 0;
            for (std::int64_t j = 0; j < p; ++j) {
                pred += (double)params_host.at(j + st) * (double)data_host.at(i, j);
            }
            if (fit_intercept) {
                pred += (double)params_host.at(0);
            }
            // We cast argument to float_t to ensure correct clipping
            double prob = clip_prob(float_t(1.0) / (float_t)(1 + std::exp(-pred)));
            logloss -=
                labels_host.at(i) * std::log(prob) + (1 - labels_host.at(i)) * std::log(1 - prob);
        }
        for (std::int64_t i = 0; i < p; ++i) {
            logloss += L1 * abs(params_host.at(i + st));
            logloss += L2 * params_host.at(i + st) * params_host.at(i + st);
        }
        return logloss;
    }

    void naive_probabilities(const ndview<float_t, 2>& data,
                             const ndview<float_t, 1>& params,
                             const ndview<std::int32_t, 1>& labels,
                             ndview<float_t, 1>& out_prob,
                             bool fit_intercept) {
        const std::int64_t n = data.get_dimension(0);
        const std::int64_t p = data.get_dimension(1);
        std::int64_t st_ind = fit_intercept;
        for (std::int64_t i = 0; i < n; ++i) {
            double pred = 0;
            for (std::int64_t j = 0; j < p; ++j) {
                pred += params.at(j + st_ind) * data.at(i, j);
            }
            if (fit_intercept) {
                pred += params.at(0);
            }
            out_prob.at(i) = clip_prob((double)1 / (1 + std::exp(-pred)));
        }
    }

    void naive_derivative(const ndview<float_t, 2>& data,
                          const ndview<float_t, 1>& probabilities,
                          const ndview<float_t, 1>& params,
                          const ndview<std::int32_t, 1>& labels,
                          ndview<float_t, 1>& out_der,
                          float_t L1,
                          float_t L2,
                          bool fit_intercept) {
        const std::int64_t n = data.get_dimension(0);
        const std::int64_t dim = params.get_dimension(0);
        for (std::int64_t j = 0; j < dim; ++j) {
            double val = 0;
            for (std::int64_t i = 0; i < n; ++i) {
                double x1;
                if (fit_intercept) {
                    x1 = j > 0 ? data.at(i, j - 1) : 1;
                }
                else {
                    x1 = data.at(i, j);
                }
                double prob = probabilities.at(i);
                val += (prob - labels.at(i)) * x1;
            }
            val += (!fit_intercept || 0 < j) ? L2 * 2 * params.at(j) : 0;
            out_der.at(j) = val;
        }
    }

    void naive_hessian(const ndview<float_t, 2>& data_host,
                       const ndview<float_t, 1>& probabilities_host,
                       ndview<float_t, 2>& out_hessian,
                       float_t L2,
                       bool fit_intercept) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);
        const std::int64_t start_ind = (fit_intercept ? 0 : 1);
        for (std::int64_t j = start_ind; j < p + 1; ++j) {
            for (std::int64_t k = start_ind; k < p + 1; ++k) {
                double val = 0;
                for (std::int64_t i = 0; i < n; ++i) {
                    double x1 = j > 0 ? data_host.at(i, j - 1) : 1;
                    double x2 = k > 0 ? data_host.at(i, k - 1) : 1;
                    double prob = probabilities_host.at(i);
                    val += x1 * x2 * (1 - prob) * prob;
                }
                out_hessian.at(j, k) = val;
            }
            if (j > 0) {
                out_hessian.at(j, j) += 2 * L2;
            }
        }
        if (!fit_intercept) {
            for (std::int64_t j = 0; j < p + 1; ++j) {
                out_hessian.at(0, j) = 0;
                out_hessian.at(j, 0) = 0;
            }
        }
    }

    void test_formula_derivative(const ndview<float_t, 2>& data,
                                 const ndview<float_t, 1>& probabilities,
                                 const ndview<float_t, 1>& params,
                                 const ndview<std::int32_t, 1>& labels,
                                 const ndview<float_t, 1>& derivative,
                                 const float_t L1,
                                 const float_t L2,
                                 bool fit_intercept,
                                 const float_t rtol = 1e-3,
                                 const float_t atol = 1e-3) {
        const std::int64_t dim = params.get_dimension(0);
        auto out_derivative =
            ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);

        naive_derivative(data,
                         probabilities,
                         params,
                         labels,
                         out_derivative,
                         L1,
                         L2,
                         fit_intercept);

        for (std::int64_t i = 0; i < dim; ++i) {
            IS_CLOSE(float_t, out_derivative.at(i), derivative.at(i), rtol, atol);
        }
    }

    void test_formula_hessian(const ndview<float_t, 2>& data,
                              const ndview<float_t, 1>& probabilities,
                              const ndview<float_t, 2>& hessian,
                              const float_t L2,
                              bool fit_intercept,
                              const float_t rtol = 1e-3,
                              const float_t atol = 1e-3) {
        const std::int64_t p = data.get_dimension(1);
        auto out_hessian =
            ndarray<float_t, 2>::empty(this->get_queue(), { p + 1, p + 1 }, sycl::usm::alloc::host);

        naive_hessian(data, probabilities, out_hessian, L2, fit_intercept);

        for (std::int64_t i = 0; i <= p; ++i) {
            for (std::int64_t j = 0; j <= p; ++j) {
                IS_CLOSE(float_t, out_hessian.at(i, j), hessian.at(i, j), rtol, atol);
            }
        }
    }

    void test_hessian_product(const ndview<float_t, 2>& hessian_host,
                              base_matrix_operator<float_t>& hessp,
                              bool fit_intercept,
                              double L2,
                              const float_t rtol = 1e-3,
                              const float_t atol = 1e-3,
                              std::int32_t num_checks = 5) {
        const std::int64_t p = hessian_host.get_dimension(0) - 1;
        const std::int64_t dim = fit_intercept ? p + 1 : p;

        auto vec_host =
            ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);

        for (std::int32_t ij = 0; ij < num_checks; ++ij) {
            primitives::host_engine eng(2007 + dim * num_checks + ij);
            pr::uniform<float_t>(dim, vec_host.get_mutable_data(), eng, -1.0, 1.0);
            auto vec_gpu = vec_host.to_device(this->get_queue());
            auto out_vector =
                ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::device);
            hessp(vec_gpu, out_vector, {}).wait_and_throw();

            auto out_vector_host = out_vector.to_host(this->get_queue());
            const std::int64_t st = fit_intercept ? 0 : 1;

            // We use double for gth computations to achieve better precision
            for (std::int64_t i = st; i < p + 1; ++i) {
                double correct = 0;
                for (std::int64_t j = st; j < p + 1; ++j) {
                    correct += static_cast<double>(vec_host.at(j - st)) *
                               static_cast<double>(hessian_host.at(i, j));
                }
                IS_CLOSE(float_t, out_vector_host.at(i - st), (float_t)correct, rtol, atol);
            }
        }
    }

    void test_functors(table& data,
                       ndview<std::int32_t, 1>& labels_gpu,
                       ndview<float_t, 1>& params_gpu,
                       ndview<float_t, 1>& gth_grad,
                       ndview<float_t, 2>& gth_hessian,
                       float_t gth_logloss,
                       const float_t L2 = 0,
                       bool fit_intercept = true,
                       bool batch_test = false,
                       const float_t rtol = 1e-3,
                       const float_t atol = 1e-3) {
        const std::int64_t p = gth_hessian.get_dimension(0) - 1;
        std::int64_t bsz = -1;
        if (batch_test) {
            bsz = GENERATE(4, 8, 16, 20, 37, 512);
        }
        // logloss_function has different regularization so we need to multiply it by 2 to align with other implementations

        auto functor = logloss_function<float_t>(this->get_queue(),
                                                 data,
                                                 labels_gpu,
                                                 L2 * 2,
                                                 fit_intercept,
                                                 bsz);
        auto set_point_event = functor.update_x(params_gpu, true, {});
        wait_or_pass(set_point_event).wait_and_throw();

        IS_CLOSE(float_t, gth_logloss, functor.get_value(), rtol, atol);
        auto grad_func = functor.get_gradient();
        auto grad_func_host = grad_func.to_host(this->get_queue());
        std::int64_t dim = fit_intercept ? p + 1 : p;

        for (std::int64_t i = 0; i < dim; ++i) {
            IS_CLOSE(float_t, gth_grad.at(i), grad_func_host.at(i), rtol, atol);
        }
        base_matrix_operator<float_t>& hessp = functor.get_hessian_product();
        test_hessian_product(gth_hessian, hessp, fit_intercept, L2, rtol, atol);
    }

protected:
    std::int64_t n_;
    std::int64_t p_;
    table data_;
    table dense_data_;
    table params_;
    ndarray<std::int32_t, 1> labels_;
};

} // namespace oneapi::dal::backend::primitives::test
