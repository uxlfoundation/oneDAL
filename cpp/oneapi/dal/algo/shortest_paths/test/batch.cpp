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

#include <array>

#include "oneapi/dal/algo/shortest_paths/traverse.hpp"
#include "oneapi/dal/graph/detail/directed_adjacency_vector_graph_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::algo::shortest_paths::test {

namespace dal = oneapi::dal;
namespace te = dal::test::engine;
namespace la = te::linalg;

constexpr double unreachable_double_distance = std::numeric_limits<double>::max();
constexpr std::int32_t unreachable_int32_t_distance = std::numeric_limits<std::int32_t>::max();

class graph_base_data {
public:
    graph_base_data() = default;

    std::int64_t get_vertex_count() const {
        return vertex_count;
    }

    std::int64_t get_edge_count() const {
        return edge_count;
    }

    std::int64_t get_cols_count() const {
        return cols_count;
    }

    std::int64_t get_rows_count() const {
        return rows_count;
    }

    std::int64_t get_source() const {
        return source;
    }

protected:
    std::int64_t vertex_count;
    std::int64_t edge_count;
    std::int64_t cols_count;
    std::int64_t rows_count;
    std::int64_t source;
};

class d_thread_buckets_graph_type : public graph_base_data {
public:
    d_thread_buckets_graph_type() {
        vertex_count = 301;
        edge_count = 600;
        cols_count = 600;
        rows_count = 302;
        source = 0;

        rows[0] = 0;
        rows[1] = 300;
        for (std::int64_t index = 2; index < rows_count; ++index) {
            rows[index] = rows[index - 1] + 1;
        }
        for (std::int64_t index = 0; index < vertex_count - 1; ++index) {
            cols[index] = index + 1;
        }
        for (int index = 1; index < vertex_count - 1; ++index) {
            cols[vertex_count + index - 2] = index + 1;
        }
        cols.back() = 1;
        for (std::int64_t index = 0; index < vertex_count - 1; ++index) {
            edge_weights[index] = (index + 1) * 10;
        }
        for (std::int64_t index = vertex_count - 1; index < cols_count; ++index) {
            edge_weights[index] = 10000;
        }
        for (int index = 0; index < vertex_count; ++index) {
            distances[index] = index * 10;
        }
    }

    std::array<std::int64_t, 302> rows;
    std::array<std::int32_t, 600> cols;
    std::array<double, 600> edge_weights;
    std::array<double, 301> distances;
};

class d_thread_bucket_size_graph_type : public graph_base_data {
public:
    d_thread_bucket_size_graph_type() {
        vertex_count = 301;
        edge_count = 600;
        cols_count = 600;
        rows_count = 302;
        source = 0;

        rows[0] = 0;
        rows[1] = vertex_count - 1;
        rows[2] = vertex_count + 1;
        for (int index = 3; index < 152; ++index) {
            rows[index] = rows[index - 1] + 1;
        }
        rows[152] = rows[151];
        for (std::int64_t index = 153; index < rows_count; ++index) {
            rows[index] = rows[index - 1] + 1;
        }
        for (int index = 1; index < vertex_count; ++index) {
            cols[index - 1] = index;
        }
        cols[vertex_count - 1] = 2;
        cols[vertex_count] = vertex_count - 1;
        std::int64_t current_index = vertex_count + 1;
        for (int index = 3; index <= 151; ++index, ++current_index) {
            cols[current_index] = index;
        }
        for (int index = 151; index <= 299; ++index, ++current_index) {
            cols[current_index] = index;
        }
        for (std::int64_t index = 0; index < 151; ++index) {
            edge_weights[index] = index * 2 + 1;
        }
        for (int index = 300; index >= 151; --index) {
            edge_weights[index] = (300 - index) * 2 + 1;
        }
        for (std::int64_t index = vertex_count; index < cols_count; ++index) {
            edge_weights[index] = 1;
        }
        for (int index = 0; index <= 151; ++index) {
            distances[index] = index;
        }
        for (int index = 152; index <= 300; ++index) {
            distances[index] = distances[index - 1] - 1;
        }
    }

    std::array<std::int64_t, 302> rows;
    std::array<std::int32_t, 600> cols;
    std::array<double, 600> edge_weights;
    std::array<double, 301> distances;
};

class d_max_element_bin_graph_type : public graph_base_data {
public:
    d_max_element_bin_graph_type() {
        vertex_count = 3001;
        edge_count = 6000;
        cols_count = 6000;
        rows_count = 3002;
        source = 0;

        rows[0] = 0;
        rows[1] = vertex_count - 1;
        rows[2] = vertex_count + 1;
        for (int index = 3; index < 1502; ++index) {
            rows[index] = rows[index - 1] + 1;
        }
        rows[1502] = rows[1501];
        for (std::int64_t index = 1503; index < rows_count; ++index) {
            rows[index] = rows[index - 1] + 1;
        }
        for (int index = 1; index < vertex_count; ++index) {
            cols[index - 1] = index;
        }
        cols[vertex_count - 1] = 2;
        cols[vertex_count] = vertex_count - 1;
        std::int64_t current_index = vertex_count + 1;
        for (int index = 3; index <= 1501; ++index, ++current_index) {
            cols[current_index] = index;
        }
        for (int index = 1501; index <= 2999; ++index, ++current_index) {
            cols[current_index] = index;
        }
        for (std::int64_t index = 0; index < 1501; ++index) {
            edge_weights[index] = index * 2 + 1;
        }
        for (int index = 3000; index >= 1501; --index) {
            edge_weights[index] = (3000 - index) * 2 + 1;
        }
        for (std::int64_t index = vertex_count; index < cols_count; ++index) {
            edge_weights[index] = 1;
        }
        for (int index = 0; index <= 1501; ++index) {
            distances[index] = index;
        }
        for (int index = 1502; index <= 3000; ++index) {
            distances[index] = distances[index - 1] - 1;
        }
    }

    std::array<std::int64_t, 3002> rows;
    std::array<std::int32_t, 6000> cols;
    std::array<double, 6000> edge_weights;
    std::array<double, 3001> distances;
};

class d_isolated_vertexes_graph_type : public graph_base_data {
public:
    d_isolated_vertexes_graph_type() {
        vertex_count = 10;
        edge_count = 0;
        cols_count = 0;
        rows_count = 11;
        source = 0;
    }
    std::array<std::int64_t, 11> rows = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::array<std::int32_t, 0> cols = {};
    std::array<double, 0> edge_weights = {};
    std::array<double, 10> distances = { 0,
                                         unreachable_double_distance,
                                         unreachable_double_distance,
                                         unreachable_double_distance,
                                         unreachable_double_distance,
                                         unreachable_double_distance,
                                         unreachable_double_distance,
                                         unreachable_double_distance,
                                         unreachable_double_distance,
                                         unreachable_double_distance };
};

class d_isolated_vertexes_int_graph_type : public graph_base_data {
public:
    d_isolated_vertexes_int_graph_type() {
        vertex_count = 10;
        edge_count = 0;
        cols_count = 0;
        rows_count = 11;
        source = 0;
    }
    std::array<std::int64_t, 11> rows = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::array<std::int32_t, 0> cols = {};
    std::array<std::int32_t, 0> edge_weights = {};
    std::array<std::int32_t, 10> distances = { 0,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance,
                                               unreachable_int32_t_distance };
};

class d_graph_3_graph_type : public graph_base_data {
public:
    d_graph_3_graph_type() {
        vertex_count = 21;
        edge_count = 49;
        cols_count = 49;
        rows_count = 22;
        source = 0;
    }
    std::array<std::int64_t, 22> rows = { 0,  3,  6,  8,  11, 14, 15, 18, 21, 22, 25,
                                          27, 30, 32, 35, 37, 39, 42, 42, 45, 47, 49 };
    std::array<std::int32_t, 49> cols = { 1,  2,  3,  2,  5,  13, 5,  6,  2,  4,  7,  0,  7,
                                          8,  6,  3,  7,  9,  6,  10, 11, 12, 5,  13, 14, 6,
                                          9,  10, 14, 16, 7,  11, 14, 17, 18, 10, 15, 11, 19,
                                          12, 15, 20, 14, 17, 19, 14, 20, 8,  15 };
    std::array<double, 49> edge_weights = { 95, 89, 70, 19, 96, 73, 42, 19, 23, 40, 10, 70, 18,
                                            64, 47, 94, 21, 22, 26, 40, 66, 91, 62, 80, 10, 57,
                                            63, 99, 73, 17, 12, 14, 42, 92, 69, 39, 58, 38, 10,
                                            87, 36, 71, 45, 93, 11, 21, 21, 66, 46 };
    std::array<double, 21> distances = { 0,   95,  89,  70,  110, 131, 106, 80,  174, 128, 120,
                                         146, 250, 168, 138, 196, 163, 260, 237, 206, 227 };
};

class d_one_bucket_graph_type : public graph_base_data {
public:
    d_one_bucket_graph_type() {
        vertex_count = 6;
        edge_count = 9;
        cols_count = 9;
        rows_count = 7;
        source = 0;
    }
    std::array<std::int64_t, 7> rows = { 0, 5, 6, 7, 8, 9, 9 };
    std::array<std::int32_t, 9> cols = { 1, 2, 3, 4, 5, 2, 3, 4, 5 };
    std::array<double, 9> edge_weights = { 1, 3, 4, 5, 6, 1, 1, 1, 1 };
    std::array<double, 6> distances = { 0, 1, 2, 3, 4, 5 };
};

class d_net_10_10_double_edges_graph_type : public graph_base_data {
public:
    d_net_10_10_double_edges_graph_type() {
        vertex_count = 100;
        edge_count = 400;
        cols_count = 400;
        rows_count = 101;
        source = 0;
    }
    std::array<std::int64_t, 101> rows = {
        0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,  56,  60,  64,
        68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112, 116, 120, 124, 128, 132,
        136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200,
        204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
        272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336,
        340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400
    };
    std::array<std::int32_t, 400> cols = {
        1,  9,  10, 90, 0,  2,  11, 91, 1,  3,  12, 92, 2,  4,  13, 93, 3,  5,  14, 94, 4,  6,  15,
        95, 5,  7,  16, 96, 6,  8,  17, 97, 7,  9,  18, 98, 0,  8,  19, 99, 0,  11, 19, 20, 1,  10,
        12, 21, 2,  11, 13, 22, 3,  12, 14, 23, 4,  13, 15, 24, 5,  14, 16, 25, 6,  15, 17, 26, 7,
        16, 18, 27, 8,  17, 19, 28, 9,  10, 18, 29, 10, 21, 29, 30, 11, 20, 22, 31, 12, 21, 23, 32,
        13, 22, 24, 33, 14, 23, 25, 34, 15, 24, 26, 35, 16, 25, 27, 36, 17, 26, 28, 37, 18, 27, 29,
        38, 19, 20, 28, 39, 20, 31, 39, 40, 21, 30, 32, 41, 22, 31, 33, 42, 23, 32, 34, 43, 24, 33,
        35, 44, 25, 34, 36, 45, 26, 35, 37, 46, 27, 36, 38, 47, 28, 37, 39, 48, 29, 30, 38, 49, 30,
        41, 49, 50, 31, 40, 42, 51, 32, 41, 43, 52, 33, 42, 44, 53, 34, 43, 45, 54, 35, 44, 46, 55,
        36, 45, 47, 56, 37, 46, 48, 57, 38, 47, 49, 58, 39, 40, 48, 59, 40, 51, 59, 60, 41, 50, 52,
        61, 42, 51, 53, 62, 43, 52, 54, 63, 44, 53, 55, 64, 45, 54, 56, 65, 46, 55, 57, 66, 47, 56,
        58, 67, 48, 57, 59, 68, 49, 50, 58, 69, 50, 61, 69, 70, 51, 60, 62, 71, 52, 61, 63, 72, 53,
        62, 64, 73, 54, 63, 65, 74, 55, 64, 66, 75, 56, 65, 67, 76, 57, 66, 68, 77, 58, 67, 69, 78,
        59, 60, 68, 79, 60, 71, 79, 80, 61, 70, 72, 81, 62, 71, 73, 82, 63, 72, 74, 83, 64, 73, 75,
        84, 65, 74, 76, 85, 66, 75, 77, 86, 67, 76, 78, 87, 68, 77, 79, 88, 69, 70, 78, 89, 70, 81,
        89, 90, 71, 80, 82, 91, 72, 81, 83, 92, 73, 82, 84, 93, 74, 83, 85, 94, 75, 84, 86, 95, 76,
        85, 87, 96, 77, 86, 88, 97, 78, 87, 89, 98, 79, 80, 88, 99, 0,  80, 91, 99, 1,  81, 90, 92,
        2,  82, 91, 93, 3,  83, 92, 94, 4,  84, 93, 95, 5,  85, 94, 96, 6,  86, 95, 97, 7,  87, 96,
        98, 8,  88, 97, 99, 9,  89, 90, 98
    };
    std::array<double, 400> edge_weights = {
        35.117, 14.866, 40.275, 33.14,  94.686, 40.318, 80.702, 79.627, 96.623, 97.797, 47.286,
        64.155, 38.207, 32.852, 69.205, 12.414, 76.32,  30.598, 33.929, 68.586, 36.309, 26.018,
        27.375, 20.903, 43.785, 18.283, 69.937, 29.769, 95.933, 16.099, 92.866, 31.05,  16.754,
        13.813, 18.274, 17.473, 86.922, 18.712, 52.721, 18.543, 49.387, 95.887, 38.343, 62.913,
        88.697, 33.67,  32.464, 26.071, 51.448, 29.814, 32.042, 57.933, 11.333, 79.42,  75.248,
        64.086, 25.613, 27.237, 23.973, 10.87,  47.463, 88.622, 96.627, 48.667, 18.191, 51.712,
        73.754, 63.582, 93.104, 41.327, 32.396, 47.937, 50.718, 41.953, 25.129, 59.559, 33.758,
        48.009, 58.5,   57.657, 48.096, 27.94,  96.489, 11.834, 14.282, 65.82,  15.01,  21.397,
        70.768, 22.85,  13.343, 24.874, 22.16,  12.014, 80.798, 99.811, 68.376, 12.51,  12.023,
        26.487, 15.549, 72.425, 15.912, 37.859, 77.267, 44.515, 90.54,  97.081, 21.26,  15.596,
        35.623, 94.747, 24.406, 41.725, 85.974, 26.192, 71.676, 25.498, 18.172, 62.532, 59.492,
        23.605, 87.549, 89.169, 70.05,  96.951, 16.093, 65.14,  77.895, 51.148, 12.266, 18.581,
        50.948, 14.375, 86.618, 98.025, 34.117, 94.285, 42.917, 76.669, 95.11,  45.264, 17.509,
        95.454, 74.06,  38.447, 69.914, 45.513, 28.274, 22.529, 34.614, 45.036, 93.782, 32.984,
        35.56,  79.572, 41.319, 54.922, 47.196, 95.591, 31.874, 61.501, 42.002, 23.613, 92.784,
        90.461, 30.046, 21.937, 75.674, 69.205, 75.173, 88.44,  44.267, 40.556, 50.023, 42.282,
        48.879, 15.516, 76.177, 94.943, 80.164, 33.457, 65.665, 60.52,  62.171, 77.257, 67.381,
        57.197, 83.234, 72.242, 57.431, 36.396, 70.998, 36.08,  89.213, 22.512, 66.308, 54.164,
        51.798, 83.848, 81.031, 65.735, 77.344, 55.19,  55.817, 45.599, 55.059, 11.927, 35.455,
        68.906, 29.122, 50.251, 87.466, 81.469, 22.931, 21.754, 71.53,  25.906, 66.043, 39.445,
        85.094, 16.797, 65.333, 64.345, 46.45,  87.058, 53.165, 27.117, 78.272, 76.22,  20.417,
        75.177, 37.1,   75.139, 67.799, 27.963, 56.964, 49.438, 76.995, 80.357, 64.174, 80.099,
        58.327, 57.681, 93.157, 34.817, 64.357, 56.06,  35.502, 67.529, 77.794, 32.049, 91.023,
        58.551, 77.075, 41.469, 95.701, 94.21,  56.975, 89.059, 87.345, 88.718, 24.934, 53.91,
        63.952, 85.342, 30.401, 72.629, 91.378, 94.888, 28.084, 87.845, 91.406, 19.85,  75.229,
        98.636, 31.755, 83.134, 28.429, 99.344, 99.462, 55.438, 59.425, 62.747, 15.774, 51.149,
        71.884, 78.122, 22.753, 55.279, 97.611, 42.588, 79.142, 94.588, 37.715, 65.559, 30.594,
        62.941, 72.438, 33.49,  60.818, 37.223, 77.107, 11.139, 88.931, 22.705, 48.409, 23.843,
        50.269, 64.342, 60.949, 93.26,  32.693, 50.269, 92.437, 94.466, 62.088, 35.645, 58.07,
        91.119, 23.388, 82.987, 60.64,  62.476, 77.89,  90.686, 36.649, 80.489, 48.693, 95.967,
        54.257, 22.809, 69.798, 52.005, 96.053, 73.839, 59.594, 49.734, 60.823, 81.327, 61.338,
        64.793, 17.442, 26.289, 87.151, 76.085, 18.407, 52.842, 88.718, 86.228, 49.705, 27.462,
        72.24,  58.157, 55.571, 25.257, 41.65,  13.043, 50.09,  74.153, 43.986, 51.126, 11.898,
        89.083, 79.128, 41.533, 69.5,   23.761, 93.06,  64.242, 69.556, 89.596, 30.209, 43.528,
        35.089, 70.38,  93.267, 39.725, 80.589, 53.035, 85.104, 91.427, 89.186, 40.414, 95.935,
        37.653, 83.659, 62.121, 51.397, 94.19,  99.911, 62.366, 53.571, 52.746, 43.035, 96.182,
        51.231, 19.98,  55.662, 72.256
    };
    std::array<double, 100> distances = { 0,
                                          35.117,
                                          75.435,
                                          166.096,
                                          198.948,
                                          190.05,
                                          146.26500000000001,
                                          50.33200000000001,
                                          33.578,
                                          14.866,
                                          40.275,
                                          115.81899999999999,
                                          122.721,
                                          154.763,
                                          230.01100000000002,
                                          186.844,
                                          135.132,
                                          93.805,
                                          51.852000000000004,
                                          67.58699999999999,
                                          103.18799999999999,
                                          131.128,
                                          146.13799999999998,
                                          159.48099999999997,
                                          240.27899999999997,
                                          201.853,
                                          157.33800000000002,
                                          141.74200000000002,
                                          111.411,
                                          125.24399999999999,
                                          115.02199999999999,
                                          138.62699999999998,
                                          154.71999999999997,
                                          166.98599999999996,
                                          253.60399999999996,
                                          231.56300000000002,
                                          193.116,
                                          170.58700000000002,
                                          137.603,
                                          173.163,
                                          204.19099999999997,
                                          203.767,
                                          173.30099999999996,
                                          248.47399999999996,
                                          298.49699999999996,
                                          315.886,
                                          238.62900000000002,
                                          215.62300000000002,
                                          217.175,
                                          245.84599999999998,
                                          227.80399999999997,
                                          225.704,
                                          230.168,
                                          259.29,
                                          282.221,
                                          348.264,
                                          295.826,
                                          252.019,
                                          239.687,
                                          188.88199999999998,
                                          189.28199999999998,
                                          161.03199999999998,
                                          194.666,
                                          272.46,
                                          315.584,
                                          351.24299999999994,
                                          300.294,
                                          205.40599999999998,
                                          185.55599999999998,
                                          157.12699999999998,
                                          89.82,
                                          145.25799999999998,
                                          171.913,
                                          247.275,
                                          284.99,
                                          290.42499999999995,
                                          267.71999999999997,
                                          203.378,
                                          153.10899999999998,
                                          95.03899999999999,
                                          66.43199999999999,
                                          86.571,
                                          123.22,
                                          177.477,
                                          268.5,
                                          289.772,
                                          213.687,
                                          161.636,
                                          103.479,
                                          53.388999999999996,
                                          33.14,
                                          45.038,
                                          68.79899999999999,
                                          158.39499999999998,
                                          228.77499999999998,
                                          210.953,
                                          176.03400000000002,
                                          81.382,
                                          51.051,
                                          33.409 };
};

class d_net_10_10_int_edges_graph_type : public graph_base_data {
public:
    d_net_10_10_int_edges_graph_type() {
        vertex_count = 100;
        edge_count = 400;
        cols_count = 400;
        rows_count = 101;
        source = 0;
    }
    std::array<std::int64_t, 101> rows = {
        0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,  56,  60,  64,
        68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112, 116, 120, 124, 128, 132,
        136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200,
        204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
        272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316, 320, 324, 328, 332, 336,
        340, 344, 348, 352, 356, 360, 364, 368, 372, 376, 380, 384, 388, 392, 396, 400
    };
    std::array<std::int32_t, 400> cols = {
        1,  9,  10, 90, 0,  2,  11, 91, 1,  3,  12, 92, 2,  4,  13, 93, 3,  5,  14, 94, 4,  6,  15,
        95, 5,  7,  16, 96, 6,  8,  17, 97, 7,  9,  18, 98, 0,  8,  19, 99, 0,  11, 19, 20, 1,  10,
        12, 21, 2,  11, 13, 22, 3,  12, 14, 23, 4,  13, 15, 24, 5,  14, 16, 25, 6,  15, 17, 26, 7,
        16, 18, 27, 8,  17, 19, 28, 9,  10, 18, 29, 10, 21, 29, 30, 11, 20, 22, 31, 12, 21, 23, 32,
        13, 22, 24, 33, 14, 23, 25, 34, 15, 24, 26, 35, 16, 25, 27, 36, 17, 26, 28, 37, 18, 27, 29,
        38, 19, 20, 28, 39, 20, 31, 39, 40, 21, 30, 32, 41, 22, 31, 33, 42, 23, 32, 34, 43, 24, 33,
        35, 44, 25, 34, 36, 45, 26, 35, 37, 46, 27, 36, 38, 47, 28, 37, 39, 48, 29, 30, 38, 49, 30,
        41, 49, 50, 31, 40, 42, 51, 32, 41, 43, 52, 33, 42, 44, 53, 34, 43, 45, 54, 35, 44, 46, 55,
        36, 45, 47, 56, 37, 46, 48, 57, 38, 47, 49, 58, 39, 40, 48, 59, 40, 51, 59, 60, 41, 50, 52,
        61, 42, 51, 53, 62, 43, 52, 54, 63, 44, 53, 55, 64, 45, 54, 56, 65, 46, 55, 57, 66, 47, 56,
        58, 67, 48, 57, 59, 68, 49, 50, 58, 69, 50, 61, 69, 70, 51, 60, 62, 71, 52, 61, 63, 72, 53,
        62, 64, 73, 54, 63, 65, 74, 55, 64, 66, 75, 56, 65, 67, 76, 57, 66, 68, 77, 58, 67, 69, 78,
        59, 60, 68, 79, 60, 71, 79, 80, 61, 70, 72, 81, 62, 71, 73, 82, 63, 72, 74, 83, 64, 73, 75,
        84, 65, 74, 76, 85, 66, 75, 77, 86, 67, 76, 78, 87, 68, 77, 79, 88, 69, 70, 78, 89, 70, 81,
        89, 90, 71, 80, 82, 91, 72, 81, 83, 92, 73, 82, 84, 93, 74, 83, 85, 94, 75, 84, 86, 95, 76,
        85, 87, 96, 77, 86, 88, 97, 78, 87, 89, 98, 79, 80, 88, 99, 0,  80, 91, 99, 1,  81, 90, 92,
        2,  82, 91, 93, 3,  83, 92, 94, 4,  84, 93, 95, 5,  85, 94, 96, 6,  86, 95, 97, 7,  87, 96,
        98, 8,  88, 97, 99, 9,  89, 90, 98
    };
    std::array<std::int32_t, 400> edge_weights = {
        84, 32,  13, 16, 13, 85, 78, 73, 73, 55, 56, 54, 36, 11, 62, 17, 25, 98, 89, 68, 22, 93,
        24, 91,  82, 27, 12, 76, 13, 41, 65, 77, 70, 60, 73, 99, 41, 31, 37, 13, 53, 63, 52, 30,
        65, 85,  80, 99, 26, 40, 23, 25, 87, 18, 94, 64, 10, 57, 13, 12, 97, 62, 41, 73, 45, 59,
        38, 24,  35, 82, 70, 62, 59, 22, 72, 71, 12, 15, 45, 92, 22, 85, 43, 42, 59, 32, 95, 21,
        24, 78,  66, 69, 39, 96, 48, 75, 19, 54, 10, 59, 62, 84, 90, 12, 73, 29, 20, 67, 46, 35,
        70, 61,  50, 92, 68, 48, 56, 36, 35, 68, 18, 16, 63, 55, 56, 29, 32, 37, 21, 37, 27, 71,
        71, 13,  67, 32, 59, 84, 52, 90, 75, 34, 43, 81, 10, 73, 70, 93, 84, 19, 18, 63, 53, 100,
        13, 83,  37, 68, 50, 74, 43, 24, 45, 28, 20, 87, 86, 98, 14, 13, 86, 63, 96, 25, 11, 93,
        50, 19,  68, 96, 32, 30, 44, 62, 34, 38, 19, 29, 54, 67, 70, 64, 65, 25, 18, 82, 36, 46,
        30, 92,  31, 69, 56, 40, 23, 75, 30, 22, 84, 52, 82, 75, 86, 70, 88, 29, 33, 91, 80, 55,
        95, 21,  69, 48, 81, 77, 49, 67, 36, 23, 51, 23, 48, 67, 39, 26, 51, 76, 34, 64, 98, 56,
        94, 94,  95, 86, 59, 42, 76, 83, 78, 95, 49, 46, 92, 75, 90, 89, 73, 51, 99, 83, 70, 19,
        28, 14,  99, 63, 51, 72, 56, 64, 27, 83, 39, 29, 63, 71, 21, 35, 60, 82, 64, 73, 99, 13,
        29, 69,  17, 78, 30, 22, 51, 63, 61, 54, 77, 70, 34, 43, 98, 29, 81, 10, 62, 76, 44, 65,
        12, 52,  24, 99, 32, 69, 85, 39, 74, 48, 81, 95, 63, 16, 72, 13, 89, 29, 45, 12, 64, 70,
        99, 100, 39, 64, 85, 29, 87, 75, 78, 81, 51, 45, 60, 13, 42, 92, 62, 59, 35, 24, 65, 61,
        38, 83,  10, 78, 88, 69, 70, 64, 74, 89, 42, 53, 87, 91, 97, 61, 44, 56, 46, 29, 51, 21,
        76, 67,  21, 81, 81, 92, 29, 76, 64, 85, 26, 24, 21, 84, 24, 68, 78, 14, 95, 38, 67, 99,
        47, 66,  96, 29
    };
    std::array<std::int32_t, 100> distances = {
        0,   84,  163, 199, 210, 228, 146, 133, 63,  32,  13,  76,  156, 179, 273, 217, 158,
        132, 110, 65,  43,  128, 154, 220, 268, 211, 182, 194, 121, 86,  85,  101, 133, 160,
        227, 223, 249, 255, 169, 148, 140, 138, 204, 192, 203, 271, 307, 240, 215, 185, 168,
        236, 266, 285, 299, 333, 299, 276, 209, 224, 208, 258, 247, 259, 346, 317, 303, 231,
        182, 221, 168, 210, 230, 208, 269, 303, 261, 219, 150, 199, 105, 121, 166, 169, 254,
        311, 219, 195, 112, 111, 16,  58,  119, 148, 215, 240, 219, 141, 74,  45
    };
};

class d_multiple_connectivity_components_graph_type : public graph_base_data {
public:
    d_multiple_connectivity_components_graph_type() {
        vertex_count = 8;
        edge_count = 6;
        cols_count = 6;
        rows_count = 9;
        source = 0;
    }
    std::array<std::int64_t, 9> rows = { 0, 3, 3, 3, 3, 6, 6, 6, 6 };
    std::array<std::int32_t, 6> cols = { 1, 2, 3, 5, 6, 7 };
    std::array<double, 6> edge_weights = { 1, 2, 3, 1, 2, 3 };
    std::array<double, 8> distances = { 0,
                                        1,
                                        2,
                                        3,
                                        unreachable_double_distance,
                                        unreachable_double_distance,
                                        unreachable_double_distance,
                                        unreachable_double_distance };
};

class d_source_isolated_vertex_graph_type : public graph_base_data {
public:
    d_source_isolated_vertex_graph_type() {
        vertex_count = 5;
        edge_count = 4;
        cols_count = 4;
        rows_count = 6;
        source = 0;
    }
    std::array<std::int64_t, 6> rows = { 0, 0, 1, 2, 3, 4 };
    std::array<std::int32_t, 4> cols = { 4, 3, 1, 2 };
    std::array<double, 4> edge_weights = { 1, 1, 1, 1 };
    std::array<double, 5> distances = { 0,
                                        unreachable_double_distance,
                                        unreachable_double_distance,
                                        unreachable_double_distance,
                                        unreachable_double_distance };
};

class d_k_15_double_edges_source_5_graph_type : public graph_base_data {
public:
    d_k_15_double_edges_source_5_graph_type() {
        vertex_count = 15;
        edge_count = 210;
        cols_count = 210;
        rows_count = 16;
        source = 5;
    }
    std::array<std::int64_t, 16> rows = { 0,   14,  28,  42,  56,  70,  84,  98,
                                          112, 126, 140, 154, 168, 182, 196, 210 };
    std::array<std::int32_t, 210> cols = {
        1, 2, 3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 0, 2, 3,  4,  5,  6,  7,
        8, 9, 10, 11, 12, 13, 14, 0, 1, 3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14,
        0, 1, 2,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 0, 1, 2,  3,  5,  6,  7,
        8, 9, 10, 11, 12, 13, 14, 0, 1, 2,  3,  4,  6,  7,  8, 9, 10, 11, 12, 13, 14,
        0, 1, 2,  3,  4,  5,  7,  8, 9, 10, 11, 12, 13, 14, 0, 1, 2,  3,  4,  5,  6,
        8, 9, 10, 11, 12, 13, 14, 0, 1, 2,  3,  4,  5,  6,  7, 9, 10, 11, 12, 13, 14,
        0, 1, 2,  3,  4,  5,  6,  7, 8, 10, 11, 12, 13, 14, 0, 1, 2,  3,  4,  5,  6,
        7, 8, 9,  11, 12, 13, 14, 0, 1, 2,  3,  4,  5,  6,  7, 8, 9,  10, 12, 13, 14,
        0, 1, 2,  3,  4,  5,  6,  7, 8, 9,  10, 11, 13, 14, 0, 1, 2,  3,  4,  5,  6,
        7, 8, 9,  10, 11, 12, 14, 0, 1, 2,  3,  4,  5,  6,  7, 8, 9,  10, 11, 12, 13
    };
    std::array<double, 210> edge_weights = {
        69, 90, 91,  83, 35, 71, 67, 25, 78, 83, 90, 26,  47, 23, 92,  61, 33, 95,  44, 63, 22,
        51, 11, 41,  33, 25, 56, 83, 39, 26, 58, 63, 100, 67, 47, 45,  27, 19, 55,  35, 51, 10,
        66, 96, 95,  36, 21, 48, 32, 60, 51, 33, 22, 84,  34, 13, 94,  50, 32, 38,  75, 64, 98,
        56, 93, 95,  54, 24, 45, 28, 60, 60, 85, 27, 99,  30, 65, 24,  69, 70, 43,  46, 66, 52,
        16, 38, 50,  94, 64, 85, 25, 53, 79, 87, 90, 34,  45, 96, 53,  54, 34, 73,  32, 47, 16,
        50, 37, 69,  81, 49, 53, 68, 61, 23, 64, 88, 35,  62, 24, 81,  41, 59, 100, 83, 26, 100,
        94, 23, 100, 62, 54, 24, 34, 92, 76, 24, 66, 24,  73, 89, 17,  45, 72, 22,  71, 19, 55,
        72, 59, 87,  63, 16, 98, 35, 53, 93, 54, 53, 10,  73, 76, 100, 12, 63, 94,  11, 47, 34,
        35, 73, 94,  99, 34, 39, 11, 39, 47, 73, 54, 93,  54, 25, 75,  95, 51, 83,  76, 18, 79,
        24, 89, 76,  24, 73, 93, 13, 11, 51, 54, 49, 23,  28, 21, 16,  98, 40, 13,  26, 11, 23
    };
    std::array<double, 15> distances = {
        46, 47, 80, 27, 53, 0, 30, 55, 24, 58, 53, 43, 46, 50, 40
    };
};

template <class T>
struct LimitedAllocator {
    typedef T value_type;

    bool is_limited = false;
    std::size_t max_allocation_size = 0;

    LimitedAllocator(bool is_limited = false, std::size_t max_allocation_size = 0)
            : is_limited(is_limited),
              max_allocation_size(max_allocation_size) {}

    template <class U>
    LimitedAllocator(const LimitedAllocator<U>& other) noexcept {
        is_limited = other.is_limited;
        max_allocation_size = other.max_allocation_size;
    }

    T* allocate(const std::size_t n) const {
        if (n == 0 || (is_limited && max_allocation_size < n)) {
            return nullptr;
        }
        if (n > static_cast<std::size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* const pv = malloc(n * sizeof(T));
        if (!pv) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(pv);
    }

    void deallocate(T* const p, std::size_t n) const noexcept {
        free(p);
    }
};

template <typename Type>
Type get_inf_value() {
    return std::numeric_limits<Type>::max();
}

class shortest_paths_test {
public:
    inline auto get_result_type(bool calculate_distances, bool calculate_predecessors) {
        using namespace dal::preview::shortest_paths;
        if (calculate_distances && calculate_predecessors) {
            return optional_results::distances | optional_results::predecessors;
        }
        else if (calculate_distances) {
            return optional_results::distances;
        }
        else if (calculate_predecessors) {
            return optional_results::predecessors;
        }
        return optional_results::distances & optional_results::predecessors;
    }

    inline bool compare_distances(std::int32_t lhs, std::int32_t rhs) {
        return lhs == rhs;
    }
    inline bool compare_distances(double lhs, double rhs) {
        const double tol = te::get_tolerance<double>(1e-4, 1e-10);
        return std::abs(lhs - rhs) < tol;
    }

    template <typename T, std::size_t Size>
    bool check_distances(const std::array<T, Size>& true_distances,
                         const std::vector<T>& distances) {
        if (true_distances.size() != distances.size()) {
            return false;
        }
        for (std::size_t index = 0; index < true_distances.size(); ++index) {
            if (!compare_distances(true_distances[index], distances[index])) {
                return false;
            }
        }
        return true;
    }

    template <typename DirectedGraphType, typename EdgeValueType, std::size_t Size>
    bool check_predecessors(const DirectedGraphType& graph,
                            const std::vector<std::int32_t>& predecessors,
                            const std::array<EdgeValueType, Size>& distances,
                            std::int64_t source) {
        EdgeValueType unreachable_distance = std::numeric_limits<EdgeValueType>::max();
        if (predecessors.size() != distances.size()) {
            return false;
        }
        if (distances[source] != 0) {
            return false;
        }
        for (std::size_t index = 0; index < predecessors.size(); ++index) {
            std::int32_t predecessor = predecessors[index];
            if (predecessor != -1) {
                oneapi::dal::preview::vertex_outward_edge_size_type<DirectedGraphType> from =
                    predecessor;
                oneapi::dal::preview::vertex_outward_edge_size_type<DirectedGraphType> to = index;
                if (!compare_distances(distances[predecessor] +
                                           oneapi::dal::preview::get_edge_value(graph, from, to),
                                       distances[index])) {
                    return false;
                }
            }
            else if (index != static_cast<std::size_t>(source)) {
                if (!compare_distances(unreachable_distance, distances[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    std::vector<T> get_data_from_table(const oneapi::dal::table& table) {
        auto arr = oneapi::dal::row_accessor<const T>(table).pull();
        const auto x = arr.get_data();
        std::vector<T> result(table.get_row_count());
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            result[i] = x[i * table.get_column_count()];
        }
        return result;
    }

    template <typename EdgeValueType, typename Allocator, std::size_t Size>
    void general_shortest_paths_check(
        const oneapi::dal::preview::directed_adjacency_vector_graph<
            std::int32_t,
            EdgeValueType,
            oneapi::dal::preview::empty_value,
            int,
            std::allocator<char>>& graph,
        double delta,
        std::int64_t source,
        oneapi::dal::preview::shortest_paths::optional_result_id result_type,
        const std::array<EdgeValueType, Size>& true_distances,
        const Allocator& alloc) {
        using namespace dal::preview::shortest_paths;
        const auto shortest_paths_desc =
            descriptor<float, method::delta_stepping, task::one_to_all, Allocator>(
                source,
                delta,
                result_type,
                Allocator(alloc));
        const auto result_shortest_paths = dal::preview::traverse(shortest_paths_desc, graph);
        if (result_type & optional_results::distances) {
            const std::vector<EdgeValueType> distances =
                get_data_from_table<EdgeValueType>(result_shortest_paths.get_distances());
            REQUIRE(check_distances(true_distances, distances));
        }
        else {
            REQUIRE_THROWS_AS(result_shortest_paths.get_distances(), uninitialized_optional_result);
        }
        if (result_type & optional_results::predecessors) {
            const std::vector<std::int32_t> predecessors =
                get_data_from_table<std::int32_t>(result_shortest_paths.get_predecessors());
            REQUIRE(check_predecessors(graph, predecessors, true_distances, source));
        }
        else {
            REQUIRE_THROWS_AS(result_shortest_paths.get_predecessors(),
                              uninitialized_optional_result);
        }
    }

    template <typename DirectedGraphType, typename EdgeValueType>
    void shortest_paths_check(double delta, bool calculate_distances, bool calculate_predecessors) {
        DirectedGraphType graph_data;
        const auto graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
            std::int32_t,
            EdgeValueType,
            oneapi::dal::preview::empty_value,
            int,
            std::allocator<char>>(graph_data.get_vertex_count(),
                                  graph_data.get_edge_count(),
                                  graph_data.rows.data(),
                                  graph_data.cols.data(),
                                  graph_data.edge_weights.data());
        const auto& graph = graph_builder.get_graph();
        std::allocator<char> alloc;
        const auto result_type = get_result_type(calculate_distances, calculate_predecessors);
        general_shortest_paths_check(graph,
                                     delta,
                                     graph_data.get_source(),
                                     result_type,
                                     graph_data.distances,
                                     alloc);
    }

    template <typename DirectedGraphType, typename EdgeValueType, typename AllocatorType>
    void shortest_paths_custom_allocator_check(double delta,
                                               bool calculate_distances,
                                               bool calculate_predecessors,
                                               AllocatorType alloc) {
        DirectedGraphType graph_data;
        const auto graph_builder = dal::preview::detail::directed_adjacency_vector_graph_builder<
            std::int32_t,
            EdgeValueType,
            oneapi::dal::preview::empty_value,
            int,
            std::allocator<char>>(graph_data.get_vertex_count(),
                                  graph_data.get_edge_count(),
                                  graph_data.rows.data(),
                                  graph_data.cols.data(),
                                  graph_data.edge_weights.data());
        const auto& graph = graph_builder.get_graph();
        const auto result_type = get_result_type(calculate_distances, calculate_predecessors);
        general_shortest_paths_check(graph,
                                     delta,
                                     graph_data.get_source(),
                                     result_type,
                                     graph_data.distances,
                                     alloc);
    }
};

#define SHORTEST_PATHS_TEST(name) TEST_M(shortest_paths_test, name, "[shortest_paths]")

SHORTEST_PATHS_TEST("Bucket count > number of threads, distances + predecessors") {
    this->shortest_paths_check<d_thread_buckets_graph_type, double>(10, true, true);
}

SHORTEST_PATHS_TEST("Bucket count > number of threads, predecessors") {
    this->shortest_paths_check<d_thread_buckets_graph_type, double>(10, false, true);
}

SHORTEST_PATHS_TEST(
    "Vertex count inside bucket > number of threads && < max_elements_in_bin, distances + predecessors") {
    this->shortest_paths_check<d_thread_bucket_size_graph_type, double>(100, true, true);
}

SHORTEST_PATHS_TEST(
    "Vertex count inside bucket > number of threads && < max_elements_in_bin, predecessors") {
    this->shortest_paths_check<d_thread_bucket_size_graph_type, double>(100, false, true);
}

SHORTEST_PATHS_TEST("Bucket size > max_elemnents_in_bin, distances + predecessors") {
    this->shortest_paths_check<d_max_element_bin_graph_type, double>(1000, true, true);
}

SHORTEST_PATHS_TEST("Bucket size > max_elemnents_in_bin, predecessors") {
    this->shortest_paths_check<d_max_element_bin_graph_type, double>(1000, false, true);
}

SHORTEST_PATHS_TEST("All vertexes are isolated, double edge weights, distances + predecessors") {
    this->shortest_paths_check<d_isolated_vertexes_graph_type, double>(15, true, true);
}

SHORTEST_PATHS_TEST("All vertexes are isolated, double edge weights, predecessors") {
    this->shortest_paths_check<d_isolated_vertexes_graph_type, double>(15, false, true);
}

SHORTEST_PATHS_TEST(
    "All vertexes are isolated, std::int32_t edge weights, distances + predecessors") {
    this->shortest_paths_check<d_isolated_vertexes_int_graph_type, std::int32_t>(15, true, true);
}

SHORTEST_PATHS_TEST("All edge weights > delta, distances + predecessors") {
    this->shortest_paths_check<d_graph_3_graph_type, double>(9, true, true);
}

SHORTEST_PATHS_TEST("All edge weights > delta, predecessors") {
    this->shortest_paths_check<d_graph_3_graph_type, double>(9, false, true);
}

SHORTEST_PATHS_TEST("All vertexes inside 1 bucket, distances + predecessors") {
    this->shortest_paths_check<d_one_bucket_graph_type, double>(10, true, true);
}

SHORTEST_PATHS_TEST("All vertexes inside 1 bucket, predecessors") {
    this->shortest_paths_check<d_one_bucket_graph_type, double>(10, false, true);
}

SHORTEST_PATHS_TEST("Custom allocator case") {
    this->shortest_paths_custom_allocator_check<d_graph_3_graph_type, double>(
        0.5,
        true,
        true,
        LimitedAllocator<char>());
}

// SHORTEST_PATHS_TEST("Custom allocator with limited allocations (cannot allocate enough memory)"){
//     this->shortest_paths_custom_allocator_check<d_graph_3_type, double>(0.5, true, true, LimitedAllocator<char>(true, 10));
// }

SHORTEST_PATHS_TEST("Double edge weights)") {
    this->shortest_paths_check<d_net_10_10_double_edges_graph_type, double>(40, true, true);
}

SHORTEST_PATHS_TEST("Int32_t edge weights)") {
    this->shortest_paths_check<d_net_10_10_int_edges_graph_type, std::int32_t>(40, true, true);
}

SHORTEST_PATHS_TEST("Calculate distances only)") {
    this->shortest_paths_check<d_net_10_10_double_edges_graph_type, double>(40, true, false);
}

SHORTEST_PATHS_TEST("Calculate predecessors only)") {
    this->shortest_paths_check<d_net_10_10_int_edges_graph_type, std::int32_t>(40, false, true);
}

SHORTEST_PATHS_TEST("Multiple connectivity components)") {
    this->shortest_paths_check<d_multiple_connectivity_components_graph_type, double>(1.5,
                                                                                      true,
                                                                                      true);
}

SHORTEST_PATHS_TEST("Isolated source vertex, distances + predecessors") {
    this->shortest_paths_check<d_source_isolated_vertex_graph_type, double>(3, true, true);
}

SHORTEST_PATHS_TEST("Isolated source vertex, predecessors") {
    this->shortest_paths_check<d_source_isolated_vertex_graph_type, double>(3, false, true);
}

SHORTEST_PATHS_TEST("Non-zero source vertex, distances + predecessors") {
    this->shortest_paths_check<d_k_15_double_edges_source_5_graph_type, double>(50, true, true);
}

SHORTEST_PATHS_TEST("Non-zero source vertex, predecessors") {
    this->shortest_paths_check<d_k_15_double_edges_source_5_graph_type, double>(50, false, true);
}

} // namespace oneapi::dal::algo::shortest_paths::test
