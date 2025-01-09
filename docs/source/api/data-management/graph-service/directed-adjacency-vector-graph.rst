.. Copyright 2021 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. _api_directed_adjacency_vector_graph_service:

========================================
Directed adjacency vector graph service
========================================

This section describes graph service functions for :ref:`api_directed_adjacency_vector_graph`.

.. list-table::
  :widths: 30 70
  :header-rows: 1

  * - Service function
    - Description
  * - :txtref:`get_vertex_count <api_get_vertex_count>`
    - Get the number of vertices in the graph
  * - :txtref:`get_edge_count <api_get_edge_count>`
    - Get the number of edges in the graph
  * - :txtref:`get_vertex_outward_degree <api_get_vertex_outward_degree>`
    - Get the outward degree for the specified vertex
  * - :txtref:`get_vertex_outward_neighbors <api_get_vertex_outward_neighbors>`
    - Get the range of the outward neighbors for the specified vertex
  * - :txtref:`get_edge_value <api_get_edge_value>`
    - Get the value of an edge represented as source and destination vertices
