/*******************************************************************************
* HDBSCAN metric validation test.
*
* Generates test datasets, runs every (metric, method) combination via oneDAL,
* prints labels in machine-readable format for comparison with sklearn.
*
* Build:
*   icpx -fsycl -I${DAALROOT}/include -L${DAALROOT}/lib/intel64 \
*       -lonedal_core -lonedal_thread -lonedal_dpc -lonedal -ltbb \
*       -o hdbscan_metric_test hdbscan_metric_test.cpp
*******************************************************************************/

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/hdbscan.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace dal = oneapi::dal;

// Generate two well-separated Gaussian clusters
std::vector<float> make_two_clusters(int n_per, int dim, float sep, float noise,
                                     int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, noise);
    std::vector<float> data(2 * n_per * dim);
    for (int i = 0; i < n_per; i++)
        for (int d = 0; d < dim; d++)
            data[i * dim + d] = gauss(rng);
    for (int i = 0; i < n_per; i++)
        for (int d = 0; d < dim; d++)
            data[(n_per + i) * dim + d] = sep + gauss(rng);
    return data;
}

// Generate three 1D clusters
std::vector<float> make_three_1d(int n_per, float sep, float noise, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, noise);
    std::vector<float> data(3 * n_per);
    for (int i = 0; i < n_per; i++) data[i] = gauss(rng);
    for (int i = 0; i < n_per; i++) data[n_per + i] = sep + gauss(rng);
    for (int i = 0; i < n_per; i++) data[2 * n_per + i] = 2 * sep + gauss(rng);
    return data;
}

// Generate cosine-separable clusters (angular separation)
std::vector<float> make_cosine_clusters(int n_per, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> angle_noise(0.0f, 0.05f);
    std::uniform_real_distribution<float> radius(5.0f, 15.0f);
    std::vector<float> data(2 * n_per * 2);
    // Cluster 0: direction ~(1, 0)
    for (int i = 0; i < n_per; i++) {
        float a = angle_noise(rng);
        float r = radius(rng);
        data[i * 2] = r * std::cos(a);
        data[i * 2 + 1] = r * std::sin(a);
    }
    // Cluster 1: direction ~(0, 1)
    for (int i = 0; i < n_per; i++) {
        float a = static_cast<float>(M_PI / 2.0) + angle_noise(rng);
        float r = radius(rng);
        data[(n_per + i) * 2] = r * std::cos(a);
        data[(n_per + i) * 2 + 1] = r * std::sin(a);
    }
    return data;
}

// High-dimensional clusters
std::vector<float> make_high_dim(int n_per, int dim, float sep, float noise,
                                 int seed) {
    return make_two_clusters(n_per, dim, sep, noise, seed);
}

struct TestCase {
    std::string name;
    std::vector<float> data;
    int64_t rows;
    int64_t cols;
    int64_t mcs;
    int64_t ms;
};

template <typename Method>
void run_test(const TestCase& tc, dal::hdbscan::distance_metric metric,
              const std::string& metric_name, const std::string& method_name,
              double degree = 2.0) {
    auto table = dal::homogen_table::wrap(tc.data.data(), tc.rows, tc.cols);

    auto desc = dal::hdbscan::descriptor<float, Method>(tc.mcs, tc.ms)
                    .set_result_options(dal::hdbscan::result_options::responses)
                    .set_metric(metric)
                    .set_degree(degree);

    try {
        const auto result = dal::compute(desc, table);

        const auto responses = result.get_responses();
        const auto acc = dal::row_accessor<const float>(responses);
        const auto labels = acc.pull({0, -1});

        std::cout << "RESULT|" << tc.name << "|" << metric_name << "|"
                  << method_name << "|" << degree << "|"
                  << result.get_cluster_count() << "|";
        for (int64_t i = 0; i < tc.rows; i++) {
            if (i > 0) std::cout << " ";
            std::cout << static_cast<int>(labels[i]);
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "ERROR|" << tc.name << "|" << metric_name << "|"
                  << method_name << "|" << degree << "|" << e.what()
                  << std::endl;
    }
}

int main() {
    // Build test cases - use the SAME seeds as the Python script
    std::vector<TestCase> tests;
    {
        auto d = make_two_clusters(50, 2, 10.0f, 0.3f, 42);
        tests.push_back({"two_clusters_2d", d, 100, 2, 10, 5});
    }
    {
        auto d = make_three_1d(20, 10.0f, 0.2f, 42);
        tests.push_back({"three_clusters_1d", d, 60, 1, 5, 5});
    }
    {
        auto d = make_high_dim(30, 10, 10.0f, 0.3f, 42);
        tests.push_back({"high_dim_10d", d, 60, 10, 10, 5});
    }

    // Cosine-specific dataset
    TestCase cosine_tc;
    {
        auto d = make_cosine_clusters(50, 42);
        cosine_tc = {"cosine_angular_2d", d, 100, 2, 10, 5};
    }

    struct MetricInfo {
        dal::hdbscan::distance_metric metric;
        std::string name;
        double degree;
        bool cosine_only;
    };

    std::vector<MetricInfo> metrics = {
        {dal::hdbscan::distance_metric::euclidean, "euclidean", 2.0, false},
        {dal::hdbscan::distance_metric::manhattan, "manhattan", 2.0, false},
        {dal::hdbscan::distance_metric::chebyshev, "chebyshev", 2.0, false},
        {dal::hdbscan::distance_metric::minkowski, "minkowski", 3.0, false},
        {dal::hdbscan::distance_metric::minkowski, "minkowski_p1", 1.0, false},
        {dal::hdbscan::distance_metric::minkowski, "minkowski_p2", 2.0, false},
        {dal::hdbscan::distance_metric::cosine, "cosine", 2.0, true},
    };

    for (const auto& mi : metrics) {
        const auto& test_cases = mi.cosine_only
            ? std::vector<TestCase>{cosine_tc}
            : tests;

        for (const auto& tc : test_cases) {
            // brute_force
            run_test<dal::hdbscan::method::brute_force>(
                tc, mi.metric, mi.name, "brute_force", mi.degree);

            // kd_tree (skip cosine - not supported)
            if (mi.metric != dal::hdbscan::distance_metric::cosine) {
                run_test<dal::hdbscan::method::kd_tree>(
                    tc, mi.metric, mi.name, "kd_tree", mi.degree);
            }
        }
    }

    return 0;
}
