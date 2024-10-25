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

.. _api_polynomial_kernel:

===================================
Polynomial kernel
===================================

.. include::  ../../../includes/kernel-functions/polynomial-kernel-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Polynomial kernel <alg_polynomial_kernel>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::polynomial_kernel`` namespace and are available via inclusion of the
``oneapi/dal/algo/polynomial_kernel.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::polynomial_kernel::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::polynomial_kernel::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::polynomial_kernel::task

.. _polynomial_kernel_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _polynomial_kernel_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::polynomial_kernel::compute_input


.. _polynomial_kernel_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::polynomial_kernel::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              polynomial_kernel::compute_result compute(const Descriptor& desc, \
                                      const polynomial_kernel::compute_input& input)

   :param desc: Polynomial Kernel algorithm descriptor :expr:`polynomial_kernel::descriptor`
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.x.is_empty == false`
      | :expr:`input.y.is_empty == false`
      | :expr:`input.x.column_count == input.y.column_count`
   Postconditions
      | :expr:`result.values.has_data == true`
      | :expr:`result.values.row_count == input.x.row_count`
      | :expr:`result.values.column_count == input.y.row_count`
