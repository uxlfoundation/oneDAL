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

.. _api_graph_service:

==============
Graph Service
==============

.. _graph_service_programming_interface:

---------------------
Programming interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::preview`` namespace and are available via inclusion of the
``oneapi/dal/graph/service_functions.hpp`` header file.


The graph service is a set of functions that allow you to get access to the
elements and characteristics of the :capterm:`graph`, such as vertex degree
or edge attribute.

Graph service functions are defined as function templates with ``Graph``
as a template parameter. Graph service functions introduce :txtref:`aliases <graph_aliases>`
to ``graph_traits`` as shown below.

.. _graph_aliases:

Related types
--------------

Aliases is a way to access :txtref:`graph <api_graph_template_type>` types using shorter notation.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Alias
     - Value
   * - ``graph_allocator<G>``
     - ``graph_traits<G>::allocator_type``
   * - ``graph_user_value_type<G>``
     - ``graph_traits<G>::graph_user_value_type``
   * - ``vertex_user_value_type<G>``
     - ``graph_traits<G>::vertex_user_value_type``
   * - ``edge_user_value_type<G>``
     - ``graph_traits<G>::edge_user_value_type``
   * - ``vertex_type<G>``
     - ``graph_traits<G>::vertex_type``
   * - ``vertex_size_type<G>``
     - ``graph_traits<G>::vertex_size_type``
   * - ``edge_size_type<G>``
     - ``graph_traits<G>::edge_size_type``
   * - ``vertex_edge_size_type<G>``
     - ``graph_traits<G>::vertex_edge_size_type``
   * - ``vertex_outward_edge_size_type<G>``
     - ``graph_traits<G>::vertex_outward_edge_size_type``
   * - ``vertex_edge_iterator_type<G>``
     - ``graph_traits<G>::vertex_edge_iterator_type``
   * - ``const_vertex_edge_iterator_type<G>``
     - ``graph_traits<G>::const_vertex_edge_iterator_type``
   * - ``vertex_edge_range_type<G>``
     - ``graph_traits<G>::vertex_edge_range_type``
   * - ``const_vertex_edge_range_type<G>``
     - ``graph_traits<G>::const_vertex_edge_range_type``
   * - ``const_vertex_outward_edge_range_type<G>``
     - ``graph_traits<G>::const_vertex_outward_edge_range_type``


Graph service functions
------------------------

Any service function has the following pattern:

.. code-block:: cpp

  template <typename Graph>
  return_type<Graph> get_[graph_element](const Graph& g, ...);



.. _api_get_vertex_count:

.. onedal_func:: oneapi::dal::preview::get_vertex_count

.. _api_get_edge_count:

.. onedal_func:: oneapi::dal::preview::get_edge_count

.. _api_get_vertex_degree:

.. onedal_func:: oneapi::dal::preview::get_vertex_degree

.. _api_get_vertex_neighbors:

.. onedal_func:: oneapi::dal::preview::get_vertex_neighbors

.. _api_get_vertex_outward_degree:

.. onedal_func:: oneapi::dal::preview::get_vertex_outward_degree

.. _api_get_vertex_outward_neighbors:

.. onedal_func:: oneapi::dal::preview::get_vertex_outward_neighbors

.. _api_get_edge_value:

.. onedal_func:: oneapi::dal::preview::get_edge_value


-------------
Usage Example
-------------

.. code-block:: cpp

  using graph_type = ...;
  const my_graph_type g = ...;
  std::cout << "The number of vertices: " << oneapi::dal::preview::get_vertex_count(g) << std::endl;
  std::cout << "The number of edges: " << oneapi::dal::preview::get_edge_count(g) << std::endl;


---------------------------------------
Service functions for supported graphs
---------------------------------------

This section contains description of service functions
supported for the specified :txtref:`graph <api_graphs>` types.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Service function
     - Valid :txtref:`graph concepts <graph_concepts>`
   * - ``get_vertex_count``
     - :capterm:`undirected graph`, :capterm:`directed graph`
   * - ``get_edge_count``
     - :capterm:`undirected graph`, :capterm:`directed graph`
   * - ``get_vertex_degree``
     - :capterm:`undirected graph`
   * - ``get_vertex_outward_degree``
     - :capterm:`directed graph`
   * - ``get_vertex_neighbors``
     - :capterm:`undirected graph`
   * - ``get_vertex_outward_neighbors``
     - :capterm:`directed graph`
   * - ``get_edge_value``
     - :capterm:`undirected graph`, :capterm:`directed graph`

.. toctree::

   graph-service/undirected-adjacency-vector-graph.rst
   graph-service/directed-adjacency-vector-graph.rst
