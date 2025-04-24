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

Linear regression is a method to model the connection between a dependent variable, which can be a vector,
and one or more explanatory variables. This is done by fitting the linear equations to the observed data.

.. |t_math| replace:: :ref:`Training <linreg_t_math>`
.. |t_norm_eq| replace:: :ref:`norm_eq <norm_eq>`
.. |t_input| replace:: :ref:`train_input <linreg_t_api_input>`
.. |t_result| replace:: :ref:`train_result <linreg_t_api_result>`
.. |t_op| replace:: :ref:`train(...) <linreg_t_api>`

.. |i_math| replace:: :ref:`Inference <linreg_i_math>`
.. |i_input| replace:: :ref:`infer_input <linreg_i_api_input>`
.. |i_result| replace:: :ref:`infer_result <linreg_i_api_result>`
.. |i_op| replace:: :ref:`infer(...) <linreg_i_api>`

.. |p_math| replace:: :ref:`Partial Training <linreg_p_math>`
.. |p_input| replace:: :ref:`partial_train_input <linreg_p_api_input>`
.. |p_result| replace:: :ref:`partial_train_result <linreg_p_api_result>`
.. |p_op| replace:: :ref:`partial_train(...) <linreg_p_api>`

.. |f_math| replace:: :ref:`Finalize Training <linreg_f_math>`
.. |f_op| replace:: :ref:`finalize_train(...) <linreg_f_api>`

=============== =========================== ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|             |t_norm_eq|          |t_op|   |t_input|   |t_result|
   |i_math|             |t_norm_eq|          |i_op|   |i_input|   |i_result|
   |p_math|             |t_norm_eq|          |p_op|   |p_input|   |p_result|
   |f_math|             |t_norm_eq|          |f_op|   |p_result|   |t_result|
=============== =========================== ======== =========== ============
