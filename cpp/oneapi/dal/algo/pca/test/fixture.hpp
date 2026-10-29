/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <cmath>
#include <vector>

#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/algo/pca/infer.hpp"
#include "oneapi/dal/algo/pca/partial_train.hpp"
#include "oneapi/dal/algo/pca/finalize_train.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/tables.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace pca = oneapi::dal::pca;

template <typename TestType, typename Derived>
class pca_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using input_t = pca::train_input<>;
    using result_t = pca::train_result<>;
    using partial_input_t = pca::partial_train_input<>;
    using partial_result_t = pca::partial_train_result<>;
    using descriptor_t = pca::descriptor<Float, Method>;

    bool not_available_on_device() {
        constexpr bool is_svd = std::is_same_v<Method, pca::method::svd>;
        return this->get_policy().is_gpu() && is_svd;
    }

    auto get_descriptor(std::int64_t component_count, bool deterministic = false) const {
        return pca::descriptor<Float, Method>{}
            .set_component_count(component_count)
            .set_deterministic(deterministic);
    }

    table get_gold_data() {
        const std::int64_t row_count = 10;
        const std::int64_t column_count = 5;
        static const Float data[] = {
            4.59,  0.81,  -1.37, -0.04, -0.75, //
            4.87,  0.34,  -0.98, 4.1,   -0.12, //
            4.44,  0.11,  -0.4,  3.27,  4.82, //
            0.59,  0.98,  -1.88, -0.64, 2.54, //
            -1.98, 2.57,  4.11,  -1.3,  -0.66, //
            3.26,  2.8,   2.65,  0.83,  2.12, //
            0.21,  4.23,  2.71,  2.2,   3.85, //
            1.27,  -1.15, 2.84,  1.11,  -1.12, //
            0.25,  1.61,  1.69,  4.51,  0.09, //
            -0.01, 0.58,  0.83,  2.73,  -1.33, //
        };
        return homogen_table::wrap(data, row_count, column_count);
    }
    table get_gold_cor() {
        const std::int64_t row_count = 5;
        const std::int64_t column_count = 5;
        static const Float data[] = {
            1.00000000,  -0.36877337, -0.60139209, 0.30105230,  0.21180972, //
            -0.36877337, 1.00000000,  0.44353396,  -0.16607387, 0.36798006, //
            -0.60139209, 0.44353396,  1.00000000,  -0.14177580, -0.14416017, //
            0.30105230,  -0.16607387, -0.14177580, 1.00000000,  0.111689627, //
            0.21180972,  0.36798006,  -0.14416017, 0.11189627,  1.00000000, //
        };
        return homogen_table::wrap(data, row_count, column_count);
    }
    table get_gold_eigenvectors() {
        const std::int64_t component_count = 5;
        static const Float data[] = {
            0.58693744,  -0.46709857, -0.57460003, 0.32053585,  0.0664451, //
            0.18207761,  0.53331191,  -0.04798588, 0.19145584,  0.80216467, //
            -0.11054327, -0.01167475, 0.38203369,  0.90339519,  -0.15991021, //
            0.77818759,  0.2819281,   0.44987364,  -0.14493743, -0.30256812, //
            -0.06750144, 0.64635716,  -0.56497445, 0.15320478,  -0.48476607, //
        };
        return homogen_table::wrap(data, component_count, component_count);
    }

    table get_gold_eigenvalues() {
        const std::int64_t component_count = 5;
        static const Float data[] = {
            2.07061706, 1.32805591, 0.88554629, 0.38030539, 0.33547535,
        };
        return homogen_table::wrap(data, 1, component_count);
    }
    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    /// Generates data whose covariance eigenvalues are proportional to `column_count - j`.
    ///
    /// @param[in] row_count    The number of rows, at least `2 * column_count + 2`
    /// @param[in] column_count The number of columns
    ///
    /// @return `row_count x column_count` table: orthogonal zero-mean cosine columns
    ///         rotated by a Householder reflection
    table get_separated_spectrum_data(std::int64_t row_count, std::int64_t column_count) const {
        REQUIRE(row_count >= 2 * column_count + 2);
        constexpr double pi = 3.14159265358979323846;
        auto data = array<Float>::empty(row_count * column_count);
        auto* x = data.get_mutable_data();

        std::vector<double> v(column_count);
        double v_norm2 = 0.0;
        for (std::int64_t j = 0; j < column_count; ++j) {
            v[j] = double(j + 1);
            v_norm2 += v[j] * v[j];
        }

        std::vector<double> b(column_count);
        for (std::int64_t i = 0; i < row_count; ++i) {
            double b_dot_v = 0.0;
            for (std::int64_t j = 0; j < column_count; ++j) {
                const double scale = std::sqrt(double(column_count - j));
                b[j] = scale * std::cos(2.0 * pi * double((j + 1) * i) / double(row_count));
                b_dot_v += b[j] * v[j];
            }
            for (std::int64_t j = 0; j < column_count; ++j) {
                x[i * column_count + j] = Float(b[j] - 2.0 * b_dot_v * v[j] / v_norm2);
            }
        }
        return homogen_table::wrap(data, row_count, column_count);
    }

    /// Checks that a result trained with `component_count` components matches the leading part
    /// of a result trained with all components.
    ///
    /// @param[in] truncated       Result trained with `component_count` components
    /// @param[in] full            Result trained with `column_count` components
    /// @param[in] component_count The number of components in `truncated`
    /// @param[in] column_count    The number of columns in the training data
    void check_truncated_matches_full(const pca::train_result<>& truncated,
                                      const pca::train_result<>& full,
                                      std::int64_t component_count,
                                      std::int64_t column_count) const {
        CAPTURE(component_count, column_count);
        const double tol = te::get_tolerance<Float>(1e-3, 1e-9);

        const auto full_eigval = row_accessor<const double>(full.get_eigenvalues()).pull();
        const auto check_leading = [&](const table& actual, const table& reference) {
            const auto act = row_accessor<const double>(actual).pull();
            const auto ref = row_accessor<const double>(reference).pull();
            REQUIRE(act.get_count() == component_count);
            for (std::int64_t i = 0; i < component_count; ++i) {
                CAPTURE(i, act[i], ref[i]);
                CHECK(std::abs(act[i] - ref[i]) <= tol * std::max(std::abs(ref[i]), 1.0));
            }
        };

        INFO("check eigenvalues");
        check_leading(truncated.get_eigenvalues(), full.get_eigenvalues());
        INFO("check singular values");
        check_leading(truncated.get_singular_values(), full.get_singular_values());
        INFO("check explained variances ratio");
        check_leading(truncated.get_explained_variances_ratio(),
                      full.get_explained_variances_ratio());

        INFO("check variances");
        const auto act_vars = row_accessor<const double>(truncated.get_variances()).pull();
        const auto ref_vars = row_accessor<const double>(full.get_variances()).pull();
        REQUIRE(act_vars.get_count() == column_count);
        for (std::int64_t j = 0; j < column_count; ++j) {
            CHECK(std::abs(act_vars[j] - ref_vars[j]) <= tol * std::abs(ref_vars[j]));
        }

        INFO("check eigenvectors up to sign");
        const auto act_vec = row_accessor<const double>(truncated.get_eigenvectors()).pull();
        const auto ref_vec = row_accessor<const double>(full.get_eigenvectors()).pull();
        REQUIRE(act_vec.get_count() == component_count * column_count);
        for (std::int64_t i = 0; i < component_count; ++i) {
            const double* a = act_vec.get_data() + i * column_count;
            const double* r = ref_vec.get_data() + i * column_count;
            double dot = 0.0;
            for (std::int64_t j = 0; j < column_count; ++j) {
                dot += a[j] * r[j];
            }
            const double sign = (dot < 0.0) ? -1.0 : 1.0;
            double diff = 0.0;
            for (std::int64_t j = 0; j < column_count; ++j) {
                diff = std::max(diff, std::abs(sign * a[j] - r[j]));
            }
            CAPTURE(i, diff);
            CHECK(diff < tol);
        }

        if (component_count < column_count) {
            INFO("check noise variance against the discarded eigenvalues of the full result");
            double total = 0.0;
            double discarded = 0.0;
            for (std::int64_t i = 0; i < column_count; ++i) {
                total += full_eigval[i];
                discarded += (i >= component_count) ? full_eigval[i] : 0.0;
            }
            const double noise_count = double(column_count - component_count);
            const double diff = std::abs(truncated.get_noise_variance() - discarded / noise_count);
            CAPTURE(truncated.get_noise_variance(), discarded / noise_count);
            CHECK(diff * noise_count <= tol * total);
        }
    }

    // TODO: add more checks, for example, check if eigenvectors are correct, checks for all possible values of normalization/whitening, check for infer gold data.
    void general_checks(const te::dataframe& data,
                        std::int64_t component_count,
                        const te::table_id& data_table_id) {
        CAPTURE(component_count);
        const table x = data.get_table(this->get_policy(), data_table_id);

        INFO("create descriptor");
        const auto pca_desc = get_descriptor(component_count);
        INFO("run training");
        const auto train_result = this->train(pca_desc, x);
        const auto model = train_result.get_model();
        check_train_result(pca_desc, data, train_result);
        INFO("run inference");
        const auto infer_result = this->infer(pca_desc, model, x);
        check_infer_result(pca_desc, model, data, infer_result);
    }

    void online_general_checks(const te::dataframe& data,
                               std::int64_t component_count,
                               const te::table_id& data_table_id,
                               std::int64_t nBlocks) {
        CAPTURE(component_count);

        const table x = data.get_table(this->get_policy(), data_table_id);

        INFO("create descriptor");
        const auto pca_desc = get_descriptor(component_count);
        INFO("run training");
        auto partial_result = dal::pca::partial_train_result();
        auto input_table = te::split_table_by_rows<float_t>(this->get_policy(), x, nBlocks);
        for (std::int64_t i = 0; i < nBlocks; ++i) {
            partial_result = this->partial_train(pca_desc, partial_result, input_table[i]);
        }
        auto train_result = this->finalize_train(pca_desc, partial_result);

        const auto model = train_result.get_model();
        check_train_result_online(pca_desc, data, train_result);
        INFO("run inference");
        const auto infer_result = this->infer(pca_desc, model, x);
        check_infer_result(pca_desc, model, data, infer_result);
    }

    void check_train_result(const pca::descriptor<Float, Method>& desc,
                            const te::dataframe& data,
                            const pca::train_result<>& result) {
        const auto [means, variances, eigenvalues, eigenvectors] = unpack_result(result);

        check_shapes(desc, data, result);
        check_nans(result);

        INFO("check if eigenvectors order is descending");
        check_eigenvalues_order(eigenvalues);

        INFO("check if eigenvectors matrix is orthogonal");
        check_eigenvectors_orthogonality(eigenvectors);

        const auto bs = te::compute_basic_statistics<double>(data);

        INFO("check if means are expected");
        check_means(bs, means);

        INFO("check if variances are expected");
        check_variances(bs, variances);
    }

    void check_train_result_online(const pca::descriptor<Float, Method>& desc,
                                   const te::dataframe& data,
                                   const pca::train_result<>& result) {
        const auto [eigenvalues, eigenvectors] = unpack_online_result(result);

        check_online_shapes(desc, data, result);
        check_nans(result);

        INFO("check if eigenvectors order is descending");
        check_eigenvalues_order(eigenvalues);

        INFO("check if eigenvectors matrix is orthogonal");
        check_eigenvectors_orthogonality(eigenvectors);
    }

    void check_infer_result(const pca::descriptor<Float, Method>& desc,
                            const pca::model<>& model,
                            const te::dataframe& data,
                            const pca::infer_result<>& result) {
        const auto eigenvectors = model.get_eigenvectors();
        const auto transformed_data = result.get_transformed_data();

        check_infer_nans(transformed_data);
        check_infer_shapes(transformed_data, eigenvectors, data);
    }

    void check_online_shapes(const pca::descriptor<Float, Method>& desc,
                             const te::dataframe& data,
                             const pca::train_result<>& result) {
        const auto [eigenvalues, eigenvectors] = unpack_online_result(result);

        const std::int64_t expected_component_count =
            (desc.get_component_count() > 0) ? desc.get_component_count() : data.get_column_count();

        INFO("check if eigenvalues shape is expected");
        REQUIRE(eigenvalues.get_row_count() == 1);
        REQUIRE(eigenvalues.get_column_count() == expected_component_count);

        INFO("check if eigenvectors shape is expected");
        REQUIRE(eigenvectors.get_row_count() == expected_component_count);
        REQUIRE(eigenvectors.get_column_count() == data.get_column_count());
    }

    void check_shapes(const pca::descriptor<Float, Method>& desc,
                      const te::dataframe& data,
                      const pca::train_result<>& result) {
        const auto [means, variances, eigenvalues, eigenvectors] = unpack_result(result);

        const std::int64_t expected_component_count =
            (desc.get_component_count() > 0) ? desc.get_component_count() : data.get_column_count();

        INFO("check if eigenvalues shape is expected");
        REQUIRE(eigenvalues.get_row_count() == 1);
        REQUIRE(eigenvalues.get_column_count() == expected_component_count);

        INFO("check if eigenvectors shape is expected");
        REQUIRE(eigenvectors.get_row_count() == expected_component_count);
        REQUIRE(eigenvectors.get_column_count() == data.get_column_count());

        INFO("check if means shape is expected");
        REQUIRE(means.get_row_count() == 1);
        REQUIRE(means.get_column_count() == data.get_column_count());

        INFO("check if variances shape is expected");
        REQUIRE(variances.get_row_count() == 1);
        REQUIRE(variances.get_column_count() == data.get_column_count());
    }

    void check_infer_shapes(const table& transformed_data,
                            const table& eigenvectors,
                            const te::dataframe& data) {
        INFO("check if transformed data shape is expected");

        REQUIRE(transformed_data.get_column_count() == eigenvectors.get_row_count());
        REQUIRE(transformed_data.get_row_count() == data.get_row_count());
    }

    void check_nans(const pca::train_result<>& result) {
        const auto [means, variances, eigenvalues, eigenvectors] = unpack_result(result);

        INFO("check if there is no NaN in eigenvalues");
        REQUIRE(te::has_no_nans(eigenvalues));

        INFO("check if there is no NaN in eigenvectors");
        REQUIRE(te::has_no_nans(eigenvectors));

        INFO("check if there is no NaN in means");
        REQUIRE(te::has_no_nans(means));

        INFO("check if there is no NaN in variances");
        REQUIRE(te::has_no_nans(variances));
    }

    void check_infer_nans(const table& transformed_data) {
        INFO("check if there is no NaN in transformed_data");
        REQUIRE(te::has_no_nans(transformed_data));
    }

    void check_eigenvalues_order(const table& eigenvalues) const {
        const auto W = la::matrix<double>::wrap(eigenvalues);
        bool is_descending = true;
        la::enumerate_linear(W, [&](std::int64_t i, double) {
            if (i > 0) {
                CAPTURE(i, W.get(i - 1), W.get(i));
                is_descending = is_descending && (W.get(i - 1) >= W.get(i));
            }
        });
        CHECK(is_descending);
    }

    void check_eigenvectors_orthogonality(const table& eigenvectors) {
        const auto V = la::matrix<double>::wrap(eigenvectors);
        const auto E = la::matrix<double>::eye(V.get_row_count());
        const auto VxVT = la::dot(V, V.t());
        const double diff = la::abs_error(VxVT, E);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        CHECK(diff < tol);
    }

    void check_means(const te::basic_statistics<double>& reference, const table& means) {
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        const double diff = te::rel_error(reference.get_means(), means, tol);
        CHECK(diff < tol);
    }

    void check_variances(const te::basic_statistics<double>& reference, const table& variances) {
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        const double diff = te::rel_error(reference.get_variances(), variances, tol);
        CHECK(diff < tol);
    }

    void check_eigenvalues(const table& reference, const table& eigenvalues) {
        const auto v_ref = la::matrix<double>::wrap(reference);
        const auto v_actual = la::matrix<double>::wrap(eigenvalues);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-6);
        const double diff = te::rel_error(v_ref, v_actual, tol);
        CHECK(diff < tol);
    }

    void check_eigenvectors(const table& reference, const table& eigenvectors) {
        const auto v_ref = la::matrix<double>::wrap(reference);
        const auto v_actual = la::matrix<double>::wrap(eigenvectors);
        const double tol = te::get_tolerance<Float>(1e-3, 1e-5);
        const double diff = te::rel_error(v_ref, v_actual, tol);
        CHECK(diff < tol);
    }

private:
    static auto unpack_online_result(const pca::train_result<>& result) {
        const auto eigenvalues = result.get_eigenvalues();
        const auto eigenvectors = result.get_eigenvectors();
        return std::make_tuple(eigenvalues, eigenvectors);
    }

    static auto unpack_result(const pca::train_result<>& result) {
        const auto means = result.get_means();
        const auto variances = result.get_variances();
        const auto eigenvalues = result.get_eigenvalues();
        const auto eigenvectors = result.get_eigenvectors();
        return std::make_tuple(means, variances, eigenvalues, eigenvectors);
    }
};

} // namespace oneapi::dal::pca::test
