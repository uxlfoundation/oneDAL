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

.. _api_linear_kernel:

=============
Linear kernel
=============

.. include::  ../../../includes/kernel-functions/linear-kernel-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Linear kernel <alg_linear_kernel>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::linear_kernel`` namespace and are available via inclusion of the
``oneapi/dal/algo/linear_kernel.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::linear_kernel::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_kernel::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_kernel::task

.. _linear_kernel_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _linear_kernel_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::linear_kernel::compute_input


.. _linear_kernel_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::linear_kernel::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              linear_kernel::compute_result compute(const Descriptor& desc, \
                                      const linear_kernel::compute_input& input)

   :param desc: Linear Kernel algorithm descriptor :expr:`linear_kernel::descriptor`.
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.is_empty == false`
