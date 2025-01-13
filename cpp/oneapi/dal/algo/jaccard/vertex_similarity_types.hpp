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

/// @file
/// Contains the definition of the input and output for Jaccard Similarity
/// algorithm

#pragma once

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/detail/vertex_similarity_types.hpp"

namespace oneapi::dal::preview::jaccard {

/// Class for the description of the input parameters of the Jaccard Similarity
/// algorithm
///
/// @tparam Graph  Type of the input graph
template <typename Graph, typename Task = task::by_default>
class vertex_similarity_input : public base {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    static_assert(detail::is_valid_graph<Graph>,
                  "Only undirected_adjacency_vector_graph is supported.");
    /// Constructs the algorithm input initialized with the graph and the caching builder.
    ///
    /// @param [in]       g        The input graph
    /// @param [in, out]  builder  The caching builder
    vertex_similarity_input(const Graph& g, caching_builder& builder);

    /// Returns the constant reference to the input graph
    const Graph& get_graph() const;

    /// Returns the caching_builder for the result
    caching_builder& get_caching_builder();

private:
    dal::detail::pimpl<detail::vertex_similarity_input_impl<Graph, Task>> impl_;
};

/// Class for the description of the result of the Jaccard Similarity algorithm
template <typename Task = task::by_default>
class vertex_similarity_result {
    static_assert(detail::is_valid_task<Task>);

public:
    using task_t = Task;
    /// Constructs the empty result
    vertex_similarity_result();

    /// Constructs the algorithm result initialized with the table of vertex pairs,
    /// the table of the corresponding computed Jaccard similarity coefficients, and
    /// the number of non-zero Jaccard similarity coefficients in the block.
    ///
    /// @param [in]   vertex_pairs        The table of size [nonzero_coeff_count x 2] with
    ///                                   vertex pairs which have non-zero Jaccard
    ///                                   similarity coefficients
    /// @param [in]   coeffs              The table of size [nonzero_coeff_count x 1] with
    ///                                   non-zero Jaccard similarity coefficients
    /// @param [in]   nonzero_coeff_count The number of non-zero Jaccard coefficients
    vertex_similarity_result(const table& vertex_pairs,
                             const table& coeffs,
                             std::int64_t nonzero_coeff_count);

    /// Returns the table of size [nonzero_coeff_count x 1] with non-zero Jaccard
    /// similarity coefficients
    const table& get_coeffs() const;

    /// Returns the table of size [nonzero_coeff_count x 2] with vertex pairs which have
    /// non-zero Jaccard similarity coefficients
    const table& get_vertex_pairs() const;

    /// The number of non-zero Jaccard similarity coefficients in the block
    std::int64_t get_nonzero_coeff_count() const;

private:
    dal::detail::pimpl<detail::vertex_similarity_result_impl> impl_;
};

template <typename Graph, typename Task>
vertex_similarity_input<Graph, Task>::vertex_similarity_input(const Graph& data,
                                                              caching_builder& builder_input)
        : impl_(new detail::vertex_similarity_input_impl<Graph, Task>(data, builder_input)) {}

template <typename Graph, typename Task>
const Graph& vertex_similarity_input<Graph, Task>::get_graph() const {
    return impl_->graph_data;
}

template <typename Graph, typename Task>
caching_builder& vertex_similarity_input<Graph, Task>::get_caching_builder() {
    return impl_->builder;
}

} // namespace oneapi::dal::preview::jaccard
