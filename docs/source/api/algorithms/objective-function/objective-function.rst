.. Copyright 2023 Intel Corporation
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

.. _api_objective_function:

==================
Objective function
==================

.. include::  ../../../includes/objective-function/objective-function-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Objective function<alg_objective_function>`.

---------------------
Programming Interface
---------------------
All types and functions in this section are declared in the
``oneapi::dal::objective_function`` namespace and be available via inclusion of the
``oneapi/dal/algo/objective_function.hpp`` header file.

Result options
--------------
.. onedal_class:: oneapi::dal::objective_function::result_option_id

Descriptor
----------
.. onedal_class:: oneapi::dal::objective_function::descriptor

Method tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::objective_function::method

Task tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::objective_function::task


.. _objective_function_c_api:

Compute :expr:`compute(...)`
-----------------------------

.. _objective_function_c_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::objective_function::compute_input

.. _objective_function_c_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::objective_function::compute_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              objective_function::compute_result compute(const Descriptor& desc, \
                                         const objective_function::compute_input& input)

   :param desc: objective_function algorithm descriptor :expr:`objective_function::descriptor`
   :param input: Input for objective_function computation. Contains data, parameters and responses

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.parameters.has_data == true`
      | :expr:`input.responses.has_data == true`
      | :expr:`input.data.row_count == input.responses.row_count`
      | :expr:`input.data.column_count + 1 == input.parameters.row_count`
      | :expr:`input.responses.column_count == 1`
      | :expr:`input.parameters.column_count == 1`
   Postconditions
     | :expr:`result.value.row_count == 1`
     | :expr:`result.value.column_count == 1`
     | :expr:`result.gradient.row_count == input.data.column_count + 1`
     | :expr:`result.gradient.column_count == 1`
     | :expr:`result.hessian.row_count == input.data.column_count + 1`
     | :expr:`result.hessian.column_count == input.data.column_count + 1`