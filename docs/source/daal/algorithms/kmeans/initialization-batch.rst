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

Batch Processing
****************

Input
+++++

Centroid initialization for K-Means clustering accepts the input
described below. Pass the ``Input ID`` as a parameter to the methods
that provide input for your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for K-Means Initialization (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table with the data to be clustered.

.. note:: The input can be an object of any class derived from ``NumericTable``.


Parameters
++++++++++

The following table lists parameters of centroid initialization
for K-Means clustering, which depend on the initialization method
parameter method.

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.15}|\Y{0.55}|

.. list-table:: Algorithm Parameters for K-Means Initialization (Batch Processing)
   :widths: 10 10 10 30
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - method
     - Default Value
     - Description
   * - ``algorithmFPType``
     - any
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - Not applicable
     - ``defaultDense``
     - Available initialization methods for K-Means clustering:

       For CPU:

       * ``defaultDense`` - uses first nClusters points as initial centroids
       * ``deterministicCSR`` - uses first nClusters points as initial centroids for data in a CSR numeric table
       * ``randomDense`` - uses random nClusters points as initial centroids
       * ``randomCSR`` - uses random nClusters points as initial centroids for data in a CSR numeric table
       * ``plusPlusDense`` - uses K-Means++ algorithm [Arthur2007]_
       * ``plusPlusCSR`` - uses K-Means++ algorithm for data in a CSR numeric table
       * ``parallelPlusDense`` - uses parallel K-Means++ algorithm [Bahmani2012]_
       * ``parallelPlusCSR`` - uses parallel K-Means++ algorithm for data in a CSR numeric table

       For GPU:

       * ``defaultDense`` - uses first nClusters points as initial centroids
       * ``randomDense`` - uses random nClusters points as initial centroids

   * - ``nClusters``
     - any
     - Not applicable
     - The number of clusters. Required.
   * - ``nTrials``
     -

       * ``parallelPlusDense``
       * ``parallelPlusCSR``

     - :math:`1`
     - The number of trails to generate all clusters but the first initial cluster. For details, see [Arthur2007]_, section 5
   * - ``oversamplingFactor``
     -

       * ``parallelPlusDense``
       * ``parallelPlusCSR``

     - :math:`0.5`
     - A fraction of nClusters in each of nRounds of parallel K-Means++. L=nClusters*oversamplingFactor points are sampled in a round. For details, see [Bahmani2012]_, section 3.3.
   * - ``nRounds``
     -

       * ``parallelPlusDense``
       * ``parallelPlusCSR``

     - :math:`5`
     - The number of rounds for parallel K-Means++. (L*nRounds) must be greater than nClusters. For details, see [Bahmani2012]_, section 3.3.
   * - ``engine``
     - any
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally for random numbers generation.


Output
++++++

Centroid initialization for K-Means clustering calculates the
result described below. Pass the ``Result ID`` as a parameter to the
methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for K-Means Initialization (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Result ID
     - Result
   * - ``centroids``
     - Pointer to the :math:`nClusters \times p` numeric table with the cluster centroids.

.. include:: ./../../includes/default_result_numeric_table.rst
