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

.. _api_rbf_kernel:

===================================
Radial Basis Function (RBF) kernel
===================================

.. include::  ../../../includes/kernel-functions/rbf-kernel-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Radial Basis Function (RBF) kernel <alg_rbf_kernel>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::rbf_kernel`` namespace and are available via inclusion of the
``oneapi/dal/algo/rbf_kernel.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::rbf_kernel::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::rbf_kernel::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::rbf_kernel::task

.. _rbf_kernel_c_api:

Training :expr:`compute(...)`
-----------------------------
.. _rbf_kernel_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::rbf_kernel::compute_input


.. _rbf_kernel_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::rbf_kernel::compute_result

Operation
~~~~~~~~~
.. function:: template <typename Descriptor> \
              rbf_kernel::compute_result compute(const Descriptor& desc, \
                                      const rbf_kernel::compute_input& input)

   :param desc: RBF Kernel algorithm descriptor :expr:`rbf_kernel::descriptor`.
   :param input: Input data for the computing operation

   Preconditions
      | :expr:`input.data.is_empty == false`
