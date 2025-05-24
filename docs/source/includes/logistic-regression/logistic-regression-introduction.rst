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

The Logistic Regression algorithm solves the :capterm:`classification` problem and predicts
class labels and probabilities of objects belonging to each class.

.. |t_math| replace:: :ref:`Training <logreg_t_math>`
.. |t_dense_batch| replace:: :ref:`dense_batch <logreg_t_math_dense_batch>`
.. |t_input| replace:: :ref:`train_input <logreg_t_api_input>`
.. |t_result| replace:: :ref:`train_result <logreg_t_api_result>`
.. |t_op| replace:: :ref:`train(...) <logreg_t_api>`

.. |i_math| replace:: :ref:`Inference <logreg_i_math>`
.. |i_dense_batch| replace:: :ref:`dense_batch <logreg_i_math_dense_batch>`
.. |i_input| replace:: :ref:`infer_input <logreg_i_api_input>`
.. |i_result| replace:: :ref:`infer_result <logreg_i_api_result>`
.. |i_op| replace:: :ref:`infer(...) <logreg_i_api>`

=============== =========================== ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|        |t_dense_batch|           |t_op|   |t_input|   |t_result|
   |i_math|        |i_dense_batch|           |i_op|   |i_input|   |i_result|
=============== =========================== ======== =========== ============
