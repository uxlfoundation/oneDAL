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

.. _api_kmeans:

=======
K-Means
=======

.. include:: ../../../includes/clustering/kmeans-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: K-Means <alg_kmeans>`.

---------------------
Programming Interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::kmeans`` namespace and be available via inclusion of the
``oneapi/dal/algo/kmeans.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::kmeans::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::kmeans::task

Model
-----
.. onedal_class:: oneapi::dal::kmeans::model


.. _kmeans_t_api:

Training :cpp:expr:`train(...)`
--------------------------------

.. _kmeans_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::kmeans::train_input

.. _kmeans_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::kmeans::train_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              kmeans::train_result train(const Descriptor& desc, \
                                         const kmeans::train_input& input)

   :param desc: K-Means algorithm descriptor :expr:`kmeans::descriptor`
   :param input: Input data for the training operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.initial_centroids.row_count == desc.cluster_count`
      | :expr:`input.initial_centroids.column_count == input.data.column_count`
   Postconditions
      | :expr:`result.labels.row_count == input.data.row_count`
      | :expr:`result.labels.column_count == 1`
      | :expr:`result.labels[i] >= 0`
      | :expr:`result.labels[i] < desc.cluster_count`
      | :expr:`result.iteration_count <= desc.max_iteration_count`
      | :expr:`result.model.centroids.row_count == desc.cluster_count`
      | :expr:`result.model.centroids.column_count == input.data.column_count`

.. _kmeans_i_api:

Inference :cpp:expr:`infer(...)`
---------------------------------

.. _kmeans_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::kmeans::infer_input

.. _kmeans_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::kmeans::infer_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              kmeans::infer_result infer(const Descriptor& desc, \
                                         const kmeans::infer_input& input)

   :param desc: K-Means algorithm descriptor :expr:`kmeans::descriptor`
   :param input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.model.centroids.has_data == true`
      | :expr:`input.model.centroids.row_count == desc.cluster_count`
      | :expr:`input.model.centroids.column_count == input.data.column_count`
   Postconditions
      | :expr:`result.labels.row_count == input.data.row_count`
      | :expr:`result.labels.column_count == 1`
      | :expr:`result.labels[i] >= 0`
      | :expr:`result.labels[i] < desc.cluster_count`


-------------
Usage Example
-------------

.. include:: ../../../includes/clustering/kmeans-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/clustering/kmeans-examples.rst
