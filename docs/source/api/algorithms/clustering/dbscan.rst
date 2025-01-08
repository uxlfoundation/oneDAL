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

.. highlight:: cpp
.. default-domain:: cpp

.. _api_dbscan:

======
DBSCAN
======

.. include:: ../../../includes/clustering/dbscan-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: DBSCAN <alg_dbscan>`.

---------------------
Programming Interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::dbscan`` namespace and are available via inclusion of the
``oneapi/dal/algo/dbscan.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::dbscan::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::dbscan::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::dbscan::task


.. _dbscan_c_api:

Computation :cpp:expr:`compute(...)`
------------------------------------

.. _dbscan_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::dbscan::compute_input

.. _dbscan_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::dbscan::compute_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              dbscan::compute_result compute(const Descriptor& desc, \
                                         const dbscan::compute_input& input)

   :param desc: DBSCAN algorithm descriptor :expr:`dbscan::descriptor`
   :param input: Input data for the compute operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`!input.weights.has_data || input.weights.row_count == input.data.row_count && input.weights.column_count == 1`

-------------
Usage Example
-------------

.. include:: ../../../includes/clustering/dbscan-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/clustering/dbscan-examples.rst
