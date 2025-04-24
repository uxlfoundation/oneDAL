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

Principal Component Analysis (PCA) is an algorithm for exploratory data analysis
and :capterm:`dimensionality reduction`. PCA transforms a set of feature vectors
of possibly correlated features to a new set of uncorrelated features, called
principal components. Principal components are the directions of the largest
variance, that is, the directions where the data is mostly spread out.

.. |t_math| replace:: :ref:`Training <pca_t_math>`
.. |t_cov| replace:: :ref:`Covariance <pca_t_math_cov>`
.. |t_svd| replace:: :ref:`SVD <pca_t_math_svd>`
.. |t_input| replace:: :ref:`train_input <pca_t_api_input>`
.. |t_result| replace:: :ref:`train_result <pca_t_api_result>`
.. |t_op| replace:: :ref:`train(...) <pca_t_api>`

.. |i_math| replace:: :ref:`Inference <pca_i_math>`
.. |i_cov| replace:: :ref:`Covariance <pca_i_math_cov>`
.. |i_svd| replace:: :ref:`SVD <pca_i_math_svd>`
.. |i_input| replace:: :ref:`infer_input <pca_i_api_input>`
.. |i_result| replace:: :ref:`infer_result <pca_i_api_result>`
.. |i_op| replace:: :ref:`infer(...) <pca_i_api>`

.. |p_math| replace:: :ref:`Partial Training <pca_p_math>`
.. |p_input| replace:: :ref:`partial_train_input <pca_p_api_input>`
.. |p_result| replace:: :ref:`partial_train_result <pca_p_api_result>`
.. |p_op| replace:: :ref:`partial_train(...) <pca_p_api>`

.. |f_math| replace:: :ref:`Finalize Training <pca_f_math>`
.. |f_op| replace:: :ref:`finalize_train(...) <pca_f_api>`

=============== ============= ============= ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |t_math|        |t_cov|       |t_svd|     |t_op|   |t_input|    |t_result|
   |i_math|        |i_cov|       |i_svd|     |i_op|   |i_input|    |i_result|
   |p_math|        |t_cov|       |t_svd|     |p_op|   |p_input|    |p_result|
   |f_math|        |t_cov|       |t_svd|     |f_op|   |p_result|   |t_result|
=============== ============= ============= ======== =========== ============
