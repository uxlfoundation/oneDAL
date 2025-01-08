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

.. _api_logreg:

===================
Logistic Regression
===================

.. include:: ../../../includes/logistic-regression/logistic-regression-introduction.rst

------------------------
Mathematical formulation
------------------------

Refer to :ref:`Developer Guide: Logistic Regression <alg_logreg>`.

---------------------
Programming Interface
---------------------

All types and functions are declared in the
``oneapi::dal::logistic_regression`` namespace and available via the inclusion of the
``oneapi/dal/algo/logistic_regression.hpp`` header file.

Result Options
--------------
.. onedal_class:: oneapi::dal::logistic_regression::result_option_id

Descriptor
----------
.. onedal_class:: oneapi::dal::logistic_regression::descriptor

Method Tags
~~~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::logistic_regression::method

Task Tags
~~~~~~~~~
.. onedal_tags_namespace:: oneapi::dal::logistic_regression::task

Model
-----
.. onedal_class:: oneapi::dal::logistic_regression::model


.. _logreg_t_api:

Training :cpp:expr:`train(...)`
--------------------------------

.. _logreg_t_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::logistic_regression::train_input

.. _logreg_t_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::logistic_regression::train_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              logistic_regression::train_result train(const Descriptor& desc, \
                                         const logistic_regression::train_input& input)

   :param desc: Logistic Regression algorithm descriptor :expr:`logistic_regression::descriptor`
   :param input: Input data for the training operation

   Preconditions
      | :expr:`input.data.has_data == true`
      | :expr:`input.responses.data.has_data == true`
      | :expr:`input.data.row_count == input.responses.row_count`
      | :expr:`input.responses.column_count == 1`
      | :expr:`desc.inverse_regularization > 0.0`
      | :expr:`desc.class_count == 2`
   Postconditions
      | :expr:`result.coefficients.row_count = 1`
      | :expr:`result.coefficients.column_count = input.data.column_count`
      | :expr:`result.intercept.row_count = 1`
      | :expr:`result.intercept.column_count = 1`
      | :expr:`result.packed_coefficients.row_count = 1`
      | :expr:`result.packed_coefficients.column_count = input.data.column_count + 1`

.. _logreg_i_api:

Inference :cpp:expr:`infer(...)`
---------------------------------

.. _logreg_i_api_input:

Input
~~~~~
.. onedal_class:: oneapi::dal::logistic_regression::infer_input

.. _logreg_i_api_result:

Result
~~~~~~
.. onedal_class:: oneapi::dal::logistic_regression::infer_result

Operation
~~~~~~~~~

.. function:: template <typename Descriptor> \
              logistic_regression::infer_result infer(const Descriptor& desc, \
                                         const logistic_regression::infer_input& input)

   :param desc: Logistic Regression algorithm descriptor :expr:`logistic_regression::descriptor`
   :param input: Input data for the inference operation

   Preconditions
      | :expr:`input.data.has_data == true`
   Postconditions
      | :expr:`result.responses.column_count == 1`
      | :expr:`result.responses.row_count == input.data.row_count`


--------
Examples
--------

.. include:: ../../../includes/logistic-regression/logistic-regression-examples.rst
