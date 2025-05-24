.. Copyright 2020 Intel Corporation
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

Introduction
************
The oneAPI Data Analytics Library (oneDAL) provides redesigned versions of interfaces that account for multi-device targets.
For example, CPU and GPU, distributed SPMD interfaces, and many more.

Algorithms Support
******************

The following algorithms are available in oneDAL:

.. tabularcolumns::  |\Y{0.5}|\Y{0.5}|

.. list-table:: Computation Modes
   :header-rows: 1
   :widths: 10 10
   :align: left

   * - Algorithm
     - Supported
   * - :ref:`alg_kmeans`
     - :ref:`batch <batch>`, :ref:`distributed <distributed>`
   * - :ref:`alg_covariance`
     - :ref:`batch <batch>`, :ref:`online <online>`, :ref:`distributed <distributed>`
   * - :ref:`alg_pca`
     - :ref:`batch <batch>`, :ref:`online <online>`, :ref:`distributed <distributed>`
   * - Logistic Regression
     - :ref:`batch <batch>`
   * - Linear Regression
     - :ref:`batch <batch>`, :ref:`online <online>`, :ref:`distributed <distributed>`
   * - :ref:`alg_df`
     - :ref:`batch <batch>`, :ref:`distributed <distributed>`
   * - Extra Trees
     - :ref:`batch <batch>`
   * - :ref:`alg_knn`
     - :ref:`batch <batch>`, :ref:`distributed <distributed>`
   * - :ref:`alg_svm`
     - :ref:`batch <batch>`
   * - :ref:`alg_dbscan`
     - :ref:`batch <batch>`, :ref:`distributed <distributed>`
   * - :ref:`alg_basic_statistics`
     - :ref:`batch <batch>`, :ref:`online <online>`, :ref:`distributed <distributed>`


.. toctree::
   :maxdepth: 1

   gpu_support.rst
   build_app/build-application.rst
   glossary.rst
   notations.rst
