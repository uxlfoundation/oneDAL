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

.. _api_basic_statistics:

================
Basic Statistics
================

.. include::  ../../../includes/statistics/basic-statistics-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Basic statistics <alg_basic_statistics>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::basic_statistics`` namespace and are available via inclusion of the
``oneapi/dal/algo/basic_statistics.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::basic_statistics::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::basic_statistics::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::basic_statistics::task

.. _basic_statistics_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _basic_statistics_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::basic_statistics::compute_input


.. _basic_statistics_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::basic_statistics::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              basic_statistics::compute_result compute(const Descriptor& desc, \
                                      const basic_statistics::compute_input& input)

   :param desc: Basic statistics algorithm descriptor :expr:`basic_statistics::descriptor`
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.is_empty == false`

.. _basic_statistics_p_api:

Partial Training
----------------
.. _basic_statistics_p_api_input:

Partial Input
~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::basic_statistics::partial_compute_input

.. _basic_statistics_p_api_result:

Partial Result and Finalize Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::basic_statistics::partial_compute_result

.. _basic_statistics_f_api:

Finalize Training
-----------------
