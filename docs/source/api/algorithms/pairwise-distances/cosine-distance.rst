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

.. _api_cosine_distance:

===============
Cosine distance
===============

.. include::  ../../../includes/pairwise-distances/cosine-distance-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Minkowski distance <alg_cosine_distance>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::cosine_distance`` namespace and are available via inclusion of the
``oneapi/dal/algo/cosine_distance.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::cosine_distance::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::cosine_distance::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::cosine_distance::task

.. _cosine_distance_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _cosine_distance_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::cosine_distance::compute_input


.. _cosine_distance_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::cosine_distance::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              cosine_distance::compute_result compute(const Descriptor& desc, \
                                      const cosine_distance::compute_input& input)

   :param desc: cosine Distance algorithm descriptor :expr:`cosine_distance::descriptor`.
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.is_empty == false`