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

.. _implicit_als_batch_computation:

Batch Processing
================

Training
********

For a description of the input and output, refer to :ref:`recommendation_system_usage_model`.

At the training stage, the implicit ALS recommender has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Training Parameters for Implicit Alternating Least Squares Computation (Batch Processing)
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Available computation methods:

       + ``defaultDense`` - performance-oriented method
       + ``fastCSR`` - performance-oriented method for CSR numeric tables

   * - ``nFactors``
     - :math:`10`
     - The total number of factors.
   * - ``maxIterations``
     - :math:`5`
     - The number of iterations.
   * - ``alpha``
     - :math:`40`
     - The rate of confidence.
   * - ``lambda``
     - :math:`0.01`
     - The parameter of the regularization.
   * - ``preferenceThreshold``
     - :math:`0`
     - Threshold used to define preference values. :math:`0` is the only threshold supported so far.

Prediction
**********

For a description of the input and output, refer to :ref:`recommendation_system_usage_model`.

At the prediction stage, the implicit ALS recommender has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Implicit Alternating Least Squares Computation (Batch Processing)
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.
