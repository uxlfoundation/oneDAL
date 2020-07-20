/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/data/accessor.hpp"
#include "oneapi/dal/data/table.hpp"

using namespace oneapi::dal;

TEST(train_kernel_lloyd_dense, test1) {
    constexpr std::int64_t row_count     = 8;
    constexpr std::int64_t column_count  = 2;
    constexpr std::int64_t cluster_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const int labels[] = { 1, 1, 1, 1, 0, 0, 0, 0 };

    const float centroids[] = { -1.5, -1.5, 1.5, 1.5 };

    const auto data_table = homogen_table{ row_count, column_count, data };

    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(cluster_count)
                                 .set_max_iteration_count(4)
                                 .set_accuracy_threshold(0.001);

    const auto result_train = train(kmeans_desc, data_table);

    ASSERT_EQ(row_count, result_train.get_labels().get_row_count());
    ASSERT_EQ(1, result_train.get_labels().get_column_count());
    const auto train_labels = row_accessor<const int>(result_train.get_labels()).pull().get_data();
    for (std::int64_t i = 0; i < row_count; ++i) {
        ASSERT_EQ(labels[i], train_labels[i]);
    }

    ASSERT_EQ(cluster_count, result_train.get_model().get_centroids().get_row_count());
    ASSERT_EQ(column_count, result_train.get_model().get_centroids().get_column_count());
    const auto train_centroids =
        row_accessor<const float>(result_train.get_model().get_centroids()).pull().get_data();
    for (std::int64_t i = 0; i < cluster_count * column_count; ++i) {
        ASSERT_FLOAT_EQ(centroids[i], train_centroids[i]);
    }

    constexpr std::int64_t infer_row_count = 9;
    const float data_infer[]               = { 1.0, 1.0,  0.0, 1.0,  1.0,  0.0,  2.0, 2.0,  7.0,
                                 0.0, -1.0, 0.0, -5.0, -5.0, -5.0, 0.0, -2.0, 1.0 };
    const auto data_infer_table = homogen_table{ infer_row_count, column_count, data_infer };

    const int infer_labels[] = { 1, 1, 1, 1, 1, 0, 0, 0, 0 };

    const auto result_test = infer(kmeans_desc, result_train.get_model(), data_infer_table);

    ASSERT_EQ(infer_row_count, result_test.get_labels().get_row_count());
    ASSERT_EQ(1, result_test.get_labels().get_column_count());
    const auto test_labels = row_accessor<const int>(result_test.get_labels()).pull().get_data();
    for (std::int64_t i = 0; i < infer_row_count; ++i) {
        ASSERT_EQ(infer_labels[i], test_labels[i]);
    }
}
