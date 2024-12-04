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

.. default-domain:: cpp

.. _api_connected_components:

====================
Connected Components
====================

.. include::  ../../../includes/graph/connected-components-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Connected Components <alg_connected_components>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::preview::connected_components`` namespace and
available via inclusion of the ``oneapi/dal/algo/connected_components.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::preview::connected_components::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::preview::connected_components::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::preview::connected_components::task

.. _connected_components_t_api:

Computing :cpp:expr:`preview::vertex_partitioning(...)`
-------------------------------------------------------

.. _connected_components_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::preview::connected_components::vertex_partitioning_input

.. _connected_components_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::preview::connected_components::vertex_partitioning_result

Operation
~~~~~~~~~

.. function:: template <typename Graph, typename Descriptor> \
              connected_components::vertex_partitioning_result preview::vertex_partitioning( \
                                         const Descriptor& desc, \
                                         const Graph& g)

   :param desc: Connected Components algorithm descriptor :expr:`connected_components::descriptor`
   :param g: Input graph

--------
Examples
--------

.. include:: ../../../includes/graph/connected-components-examples.rst
