.. Copyright 2020 Intel Corporation
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

.. highlight:: cpp

.. _dm_graphs:

======
Graphs
======

This section describes the types and functions related to the :txtref:`graph` concept.

.. _graph_concepts:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Type
     - Description

   * - :ref:`api_undirected_adjacency_vector_graph`
     - An implementation of the :capterm:`undirected graph` concept.

   * - :ref:`api_directed_adjacency_vector_graph`
     - An implementation of the :capterm:`directed graph` concept.

   * - :ref:`api_graph_traits`
     - A standartized way to access various properties of the :capterm:`graph`.

---------------------------
Requirements on graph types
---------------------------

Each implementation of :txtref:`graph` concept:

1. Follows the definition of the :txtref:`graph` concept and its restrictions
   (for example, :capterm:`immutability`)

2. Is :term:`reference-counted <Reference-counted object>`

3. Defines :txtref:`graph_traits <api_graph_traits>` data type.

-----------
Graph types
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Graph type
     - Description

   * - :txtref:`undirected_adjacency_vector_graph`
     - A sparse :capterm:`undirected <Undirected graph>` :capterm:`weighted <Weighted graph>`
       or :capterm:`unweighted <Unweighted graph>` graph that contains :capterm:`graph` in
       :term:`CSR <CSR data>` data format.

   * - :txtref:`directed_adjacency_vector_graph`
     - A sparse :capterm:`directed <Directed graph>` :capterm:`weighted <Weighted graph>`
       or :capterm:`unweighted <Unweighted graph>` graph that contains :capterm:`graph` in
       :term:`CSR <CSR data>` data format.

.. toctree::
   :hidden:

   graph/undirected-adjacency-vector-graph.rst
   graph/directed-adjacency-vector-graph.rst
