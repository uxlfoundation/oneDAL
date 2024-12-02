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

The K-Means initialization algorithm receives :math:`n` feature vectors as input
and chooses :math:`k` initial centroids. After initialization, K-Means algorithm
uses the initialization result to partition input data into :math:`k` clusters.

.. |c_math| replace:: :ref:`Computing <kmeans_init_c_math>`
.. |c_dense| replace:: :ref:`Dense <kmeans_init_c_math_dense>`
.. |c_random_dense| replace:: :ref:`Random dense <kmeans_init_c_math_random_dense>`
.. |c_plus_plus_dense| replace:: :ref:`K-Means++ <kmeans_init_c_math_plus_plus_dense>`
.. |c_parallel_plus_dense| replace:: :ref:`K-Means++ parallel <kmeans_init_c_math_parallel_plus_dense>`
.. |c_input| replace:: :ref:`compute_input(...) <kmeans_init_c_api_input>`
.. |c_result| replace:: :ref:`compute_result(...) <kmeans_init_c_api_result>`
.. |c_op| replace:: :ref:`compute(...) <kmeans_init_c_api>`

=============== =========== ================== ===================== ========================= ======== =========== ============
 **Operation**                            **Computational methods**                               **Programming Interface**
--------------- ------------------------------------------------------------------------------ ---------------------------------
   |c_math|      |c_dense|   |c_random_dense|   |c_plus_plus_dense|   |c_parallel_plus_dense|   |c_op|   |c_input|   |c_result|
=============== =========== ================== ===================== ========================= ======== =========== ============
