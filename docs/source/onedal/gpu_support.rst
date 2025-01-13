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

.. _onedal_gpu_support:

CPU and GPU Support
===================
Not all computation modes, methods, and parameters are supported on both CPU and GPU.
See the differences in CPU and GPU support below.

GPU-Supported Targets
*********************
The oneAPI Data Analytics Library (oneDAL) is designed to work with Intel(R) GPUs specifically, but it could potentially
run on other hardware platforms if a SYCL runtime is available.

.. tabularcolumns::  |\Y{0.5}|\Y{0.5}|

.. list-table:: GPU Support Considerations
   :header-rows: 1
   :widths: 10 10
   :align: left

   * - Algorithm
     - GPU Support
   * - :ref:`alg_kmeans`
     -
   * - :ref:`alg_covariance`
     -
   * - :ref:`alg_pca`
     -
   * - Logistic Regression
     -
   * - Linear Regression
     -
   * - :ref:`alg_df`
     -
   * - Extra Trees
     -
   * - :ref:`alg_knn`
     -
   * - :ref:`alg_svm`
     -
   * - :ref:`alg_dbscan`
     -
   * - :ref:`alg_basic_statistics`
     -
