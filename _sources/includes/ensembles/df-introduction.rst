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

Decision Forest (DF) :capterm:`classification` and :capterm:`regression` algorithms are based on an ensemble of
tree-structured classifiers, which are known as :ref:`decision trees <dt>`. Decision forest is built
using the general technique of bagging, a bootstrap aggregation, and a random choice of features.
For more details, see [Breiman84]_ and [Breiman2001]_.

.. |t_math| replace:: :ref:`Training <df_t_math>`
.. |t_dense| replace:: :ref:`Dense <df_t_math_dense>`
.. |t_hist| replace:: :ref:`Hist <df_t_math_hist>`
.. |t_input| replace:: :ref:`train_input <df_t_api_input>`
.. |t_result| replace:: :ref:`train_result <df_t_api_result>`
.. |t_op| replace:: :ref:`train(...) <df_t_api>`

.. |i_math| replace:: :ref:`Inference <df_i_math>`
.. |i_dense| replace:: :ref:`Dense <df_i_math_dense_hist>`
.. |i_hist| replace:: :ref:`Hist <df_i_math_dense_hist>`
.. |i_input| replace:: :ref:`infer_input <df_i_api_input>`
.. |i_result| replace:: :ref:`infer_result <df_i_api_result>`
.. |i_op| replace:: :ref:`infer(...) <df_i_api>`

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|        |t_dense|     |t_hist|    |t_op|   |t_input|   |t_result|
   |i_math|        |i_dense|     |i_hist|    |i_op|   |i_input|   |i_result|
=============== ============= ============= ======== =========== ============
