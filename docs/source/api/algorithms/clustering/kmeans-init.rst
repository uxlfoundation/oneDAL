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
.. default-domain:: cpp

.. _api_kmeans_init:

======================
K-Means initialization
======================

.. include:: ../../../includes/clustering/kmeans-init-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: K-Means Initialization <alg_kmeans_init>`.

---------------------
Programming Interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::kmeans_init`` namespace and be available via inclusion of the
``oneapi/dal/algo/kmeans_init.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::kmeans_init::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans_init::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans_init::task

Computing :cpp:expr:`compute(...)`
--------------------------------------

.. _kmeans_init_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::kmeans_init::compute_input

.. _kmeans_init_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::kmeans_init::compute_result

.. _kmeans_init_c_api:

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              kmeans_init::compute_result compute(const Descriptor& desc, \
                                         const kmeans_init::compute_input& input)

   :param desc: K-Means algorithm descriptor :expr:`kmeans_init::descriptor`
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.data.row_count == desc.cluster_count`
   Postconditions
      | :expr:`result.centroids.has_data == true`
      | :expr:`result.centroids.row_count == desc.cluster_count`
      | :expr:`result.centroids.column_count == input.data.column_count`


-------------
Usage Example
-------------

.. include:: ../../../includes/clustering/kmeans-init-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/clustering/kmeans-init-examples.rst
