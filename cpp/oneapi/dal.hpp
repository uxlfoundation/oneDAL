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

#pragma once

/* Common */
#include "oneapi/dal/array.hpp"
#include "oneapi/dal/chunked_array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/read.hpp"
#include "oneapi/dal/train.hpp"
#include "oneapi/dal/partial_compute.hpp"
#include "oneapi/dal/finalize_compute.hpp"
#include "oneapi/dal/partial_train.hpp"
#include "oneapi/dal/finalize_train.hpp"

/* Tables */
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/column_accessor.hpp"

/* Graphs */
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/graph/directed_adjacency_vector_graph.hpp"

/* I/O */
#include "oneapi/dal/io/csv.hpp"

/* Algos */
#include "oneapi/dal/algo/connected_components.hpp"
#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/algo/jaccard.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/algo/linear_kernel.hpp"
#include "oneapi/dal/algo/louvain.hpp"
#include "oneapi/dal/algo/objective_function.hpp"
#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/algo/polynomial_kernel.hpp"
#include "oneapi/dal/algo/sigmoid_kernel.hpp"
#include "oneapi/dal/algo/rbf_kernel.hpp"
#include "oneapi/dal/algo/shortest_paths.hpp"
#include "oneapi/dal/algo/svm.hpp"
#include "oneapi/dal/algo/triangle_counting.hpp"
#include "oneapi/dal/algo/basic_statistics.hpp"
#include "oneapi/dal/algo/finiteness_checker.hpp"
