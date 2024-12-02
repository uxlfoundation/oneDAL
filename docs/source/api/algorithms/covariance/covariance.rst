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

.. _api_covariance:

================
Covariance
================

.. include::  ../../../includes/covariance/covariance-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Covariance <alg_covariance>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::covariance`` namespace and are available via inclusion of the
``oneapi/dal/algo/covariance.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::covariance::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::covariance::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::covariance::task

.. _covariance_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _covariance_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::covariance::compute_input

.. _covariance_c_api_result:

Result and Finalize Result
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::covariance::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              covariance::compute_result compute(const Descriptor& desc, \
                                      const covariance::compute_input& input)

   :param desc: Covariance algorithm descriptor :expr:`covariance::descriptor`
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.is_empty == false`

.. _covariance_p_api:

Partial Training
----------------
.. _covariance_p_api_input:

Partial Input
~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::covariance::partial_compute_input

.. _covariance_p_api_result:

Partial Result and Finalize Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::covariance::partial_compute_result

.. _covariance_f_api:

Finalize Training
-----------------

-------------
Usage Example
-------------

.. include:: ../../../includes/covariance/covariance-usage-examples.rst