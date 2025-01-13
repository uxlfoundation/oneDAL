.. Copyright 2024 Intel Corporation
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

.. _api_linear-regression:

=================
Linear Regression
=================

.. include::  ../../../includes/linear-regression/linear-regression-introduction.rst

------------------------
Mathematical Formulation
------------------------

Refer to :ref:`Developer Guide: Linear Regression <alg_linear-regression>`.

---------------------
Programming Interface
---------------------

All types and functions from this section are declared in the
``oneapi::dal::linear_regression`` namespace and available by including the
``oneapi/dal/algo/linear_regression.hpp`` header file.

Descriptor
----------
.. onedal_class:: oneapi::dal::linear_regression::descriptor

Method Tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_regression::method

Task Tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::linear_regression::task

.. _linreg_t_api:

Training
--------
.. _linreg_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::linear_regression::train_input

.. _linreg_t_api_result:

Result and Finalize Result
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::linear_regression::train_result

.. _linreg_p_api:

Partial Training
----------------
.. _linreg_p_api_input:

Partial Input
~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::linear_regression::partial_train_input

.. _linreg_p_api_result:

Partial Result and Finalize Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. onedal_class:: oneapi::dal::linear_regression::partial_train_result

.. _linreg_f_api:

Finalize Training
-----------------

.. _linreg_i_api:

Inference
---------
.. _linreg_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::linear_regression::infer_input

.. _linreg_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::linear_regression::infer_result
