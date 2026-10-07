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

.. highlight:: cpp
.. default-domain:: cpp

.. _api_hdbscan:

=======
HDBSCAN
=======

.. include:: ../../../includes/clustering/hdbscan-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: HDBSCAN <alg_hdbscan>`.

---------------------
Programming Interface
---------------------

All types and functions in this section are declared in the
``oneapi::dal::hdbscan`` namespace and are available via inclusion of the
``oneapi/dal/algo/hdbscan.hpp`` header file.

Enum classes
------------
.. onedal_enumclass:: oneapi::dal::hdbscan::distance_metric
.. onedal_enumclass:: oneapi::dal::hdbscan::cluster_selection_method
.. onedal_enumclass:: oneapi::dal::hdbscan::store_centers_method

Result options
--------------
.. onedal_class:: oneapi::dal::hdbscan::result_option_id

Descriptor
----------
.. onedal_class:: oneapi::dal::hdbscan::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::hdbscan::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::hdbscan::task


.. _hdbscan_c_api:

Computation :cpp:expr:`compute(...)`
------------------------------------

.. _hdbscan_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::hdbscan::compute_input

.. _hdbscan_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::hdbscan::compute_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              hdbscan::compute_result compute(const Descriptor& desc, \
                                              const hdbscan::compute_input& input)

   :param desc: HDBSCAN algorithm descriptor :expr:`hdbscan::descriptor`
   :param input: Input data for the compute operation

   Preconditions
      | :expr:`input.data.has_data == true`

--------
Examples
--------

.. include:: ../../../includes/clustering/hdbscan-examples.rst
