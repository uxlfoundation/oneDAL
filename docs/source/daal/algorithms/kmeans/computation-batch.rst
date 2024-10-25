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

.. _kmeans_computation_batch:

Batch Processing
****************

Algorithm Input
+++++++++++++++

The K-Means clustering algorithm accepts the input described
below. Pass the ``Input ID`` as a parameter to the methods that
provide input for your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for K-Means Computation (Batch Processing)
   :header-rows: 1
   :widths: 10 60
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table with the data to be clustered.
   * - ``inputCentroids``
     - Pointer to the :math:`nClusters \times p` numeric table with the initial centroids.

.. note:: The input for ``data`` and ``inputCentroids`` can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
++++++++++++++++++++

The K-Means clustering algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for K-Means Computation (Batch Processing)
   :header-rows: 1
   :widths: 10 10 60
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
     - Available computation methods for K-Means clustering:

       For CPU:

       - ``defaultDense`` - implementation of Lloyd's algorithm
       - ``lloydCSR`` - implementation of Lloyd's algorithm for CSR numeric tables

       For GPU:

       - ``defaultDense`` - implementation of Lloyd's algorithm

   * - ``nClusters``
     - Not applicable
     - The number of clusters. Required to initialize the algorithm.
   * - ``maxIterations``
     - Not applicable
     - The number of iterations. Required to initialize the algorithm.
   * - ``accuracyThreshold``
     - :math:`0.0`
     - The threshold for termination of the algorithm.
   * - ``gamma``
     - :math:`1.0`
     - The weight to be used in distance calculation for binary categorical features.
   * - ``distanceType``
     - ``euclidean``
     - The measure of closeness between points (observations) being clustered. The only distance type supported so far is the Euclidean distance.
   * - **DEPRECATED:** ``assignFlag``

       **USE INSTEAD:** ``resultsToEvaluate``

     - ``true``
     - A flag that enables computation of assignments, that is, assigning cluster indices to respective observations.
   * - ``resultsToEvaluate``
     - ``computeCentroids`` | ``computeAssignments`` | ``computeExactObjectiveFunction``
     - The 64-bit integer flag that specifies which extra characteristics of the K-Means algorithm to compute.

       Provide one of the following values to request a single characteristic or use bitwise OR to request a combination of the characteristics:

       - ``computeCentroids`` for computation centroids.
       - ``computeAssignments`` for computation of assignments, that is, assigning cluster indices to respective observations.
       - ``computeExactObjectiveFunction`` for computation of exact ObjectiveFunction.


Algorithm Output
++++++++++++++++

The K-Means clustering algorithm calculates the result described
below. Pass the ``Result ID`` as a parameter to the methods that access
the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for K-Means Computation (Batch Processing)
   :header-rows: 1
   :widths: 10 60
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``centroids``
     -
       Pointer to the :math:`nClusters \times p` numeric table with the cluster centroids,
       computed when ``computeCentroids`` option is enabled.

       .. include:: ./../../includes/default_result_numeric_table.rst

   * - ``assignments``
     -
       Pointer to the :math:`n \times 1` numeric table with
       assignments of cluster indices to feature vectors in the input data,
       computed when ``computeAssignments`` option is enabled.

       .. include:: ./../../includes/default_result_numeric_table.rst

   * - ``objectiveFunction``
     -
       Pointer to the :math:`1 \times 1` numeric table with the minimum value of the objective function
       obtained at the last iteration of the algorithm, might be inexact.
       When ``computeExactObjectiveFunction`` option is enabled, exact objective function is computed.

       .. include:: ./../../includes/default_result_numeric_table.rst

   * - ``nIterations``
     -
       Pointer to the :math:`1 \times 1` numeric table with the actual number of iterations
       done by the algorithm.

       .. include:: ./../../includes/default_result_numeric_table.rst

.. note::
  You can skip update of centroids and objectiveFunction in the
  result and compute assignments using original inputCentroids.
  To do this, set ``resultsToEvaluate`` flag only to ``computeAssignments`` and ``maxIterations`` to zero.
