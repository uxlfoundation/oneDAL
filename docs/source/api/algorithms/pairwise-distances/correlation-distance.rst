.. Copyright contributors to the oneDAL project
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

.. _api_correlation_distance:

===================================
Correlation distance
===================================

.. include::  ../../../includes/pairwise-distances/correlation-distance-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Correlation distance <alg_correlation_distance>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::correlation_distance`` namespace and are available via inclusion of the
``oneapi/dal/algo/correlation_distance.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::correlation_distance::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::correlation_distance::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::correlation_distance::task

.. _correlation_distance_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _correlation_distance_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::correlation_distance::compute_input


.. _correlation_distance_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::correlation_distance::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              correlation_distance::compute_result compute(const Descriptor& desc, \
                                      const correlation_distance::compute_input& input)

   :param desc: Correlation Distance algorithm descriptor :expr:`correlation_distance::descriptor`.
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.is_empty == false`
