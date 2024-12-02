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

.. _api_subgraph_isomorphism:

====================
Subgraph Isomorphism
====================

.. include::  ../../../includes/graph/subgraph-isomorphism-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Subgraph Isomorphism <alg_subgraph_isomorphism>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::preview::subgraph_isomorphism`` namespace and
available via inclusion of the ``oneapi/dal/algo/subgraph_isomorphism.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::preview::subgraph_isomorphism::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::preview::subgraph_isomorphism::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::preview::subgraph_isomorphism::task

Enum classes
~~~~~~~~~~~~
.. onedal_enumclass:: oneapi::dal::preview::subgraph_isomorphism::kind

.. _subgraph_isomorphism_t_api:

Computing :cpp:expr:`preview::graph_matching(...)`
--------------------------------------------------

.. _subgraph_isomorphism_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::preview::subgraph_isomorphism::graph_matching_input

.. _subgraph_isomorphism_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::preview::subgraph_isomorphism::graph_matching_result

Operation
~~~~~~~~~

.. function:: template <typename Graph, typename Descriptor> \
              subgraph_isomorphism::graph_matching_result preview::graph_matching( \
                                         const Descriptor& desc, \
                                         const Graph& target, \
                                         const Graph& pattern)

   :param desc: Subgraph Isomorphism algorithm descriptor :expr:`subgraph_isomorphism::descriptor`
   :param target:  Target (big) graph
   :param pattern: Pattern (small) graph

--------
Examples
--------

.. include:: ../../../includes/graph/subgraph-isomorphism-examples.rst
