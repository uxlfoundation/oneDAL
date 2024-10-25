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

The K-Means algorithm solves :capterm:`clustering` problem by partitioning
:math:`n` feature vectors into :math:`k` clusters minimizing some criterion.
Each cluster is characterized by a representative point, called *a centroid*.

.. |t_math| replace:: :ref:`Training <kmeans_t_math>`
.. |t_lloyd| replace:: :ref:`Lloyd's <kmeans_t_math_lloyd>`
.. |t_input| replace:: :ref:`train_input <kmeans_t_api_input>`
.. |t_result| replace:: :ref:`train_result <kmeans_t_api_result>`
.. |t_op| replace:: :ref:`train(...) <kmeans_t_api>`

.. |i_math| replace:: :ref:`Inference <kmeans_i_math>`
.. |i_lloyd| replace:: :ref:`Lloyd's <kmeans_i_math_lloyd>`
.. |i_input| replace:: :ref:`infer_input <kmeans_i_api_input>`
.. |i_result| replace:: :ref:`infer_result <kmeans_i_api_result>`
.. |i_op| replace:: :ref:`infer(...) <kmeans_i_api>`

=============== =========================== ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|             |t_lloyd|            |t_op|   |t_input|   |t_result|
   |i_math|             |i_lloyd|            |i_op|   |i_input|   |i_result|
=============== =========================== ======== =========== ============
