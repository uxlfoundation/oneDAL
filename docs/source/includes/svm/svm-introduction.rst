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

Support Vector Machine (SVM) :capterm:`classification` and :capterm:`regression` are among popular
algorithms. It belongs to a family of generalized linear classification problems.


.. |t_math| replace::    :ref:`Training <svm_t_math>`
.. |t_thunder| replace:: :ref:`Thunder <svm_t_math_thunder>`
.. |t_smo| replace::     :ref:`SMO <svm_t_math_smo>`
.. |t_input| replace::   :ref:`train_input <svm_t_api_input>`
.. |t_result| replace::  :ref:`train_result <svm_t_api_result>`
.. |t_op| replace::      :ref:`train(...) <svm_t_api>`

.. |i_math| replace::    :ref:`Inference <svm_i_math>`
.. |i_smo| replace::     :ref:`SMO <svm_i_math_smo>`
.. |i_thunder| replace:: :ref:`Thunder <svm_i_math_thunder>`
.. |i_input| replace::   :ref:`infer_input <svm_i_api_input>`
.. |i_result| replace::  :ref:`infer_result <svm_i_api_result>`
.. |i_op| replace::      :ref:`infer(...) <svm_i_api>`

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|        |t_smo|     |t_thunder|    |t_op|   |t_input|   |t_result|
   |i_math|        |i_smo|     |i_thunder|    |i_op|   |i_input|   |i_result|
=============== ============= ============= ======== =========== ============
