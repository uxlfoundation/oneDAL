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
================

Algorithm Parameters
********************

The DBSCAN clustering algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for DBSCAN (Batch Processing)
   :widths: 10 10 60
   :header-rows: 1
   :class: longtable

   * - Parameter
     - Default Valude
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Available methods for computation of DBSCAN algorithm:

       - ``defaultDense`` – uses brute-force for neighborhood computation

   * - ``epsilon``
     - Not applicable
     - The maximum distance between observations lying in the same neighborhood.
   * - ``minObservations``
     - Not applicable
     - The number of observations in a neighborhood for an observation to be considered as a :term:`core <core observation>` one.
   * - ``memorySavingMode``
     - ``false``
     - If flag is set to false, all neighborhoods will be computed and stored prior to clustering.
       It will require up to :math:`O(|\text{sum of sizes of all observations' neighborhoods}|)` of additional memory,
       which in worst case can be :math:`O(|\text{number of observations}|^2)`. However, in general, performance may be better.

       .. note::
          On GPU, the ``memorySavingMode`` flag can only be set to ``true``.
          You will get an error if the flag is set to ``false``.

   * - ``resultsToCompute``
     - :math:`0`
     - The 64-bit integer flag that specifies which extra characteristics of the DBSCAN algorithm to compute.

       Provide one of the following values to request a single characteristic or
       use bitwise OR to request a combination of the characteristics:

       - ``computeCoreIndices`` for indices of core observations
       - ``computeCoreObservations`` for core observations

Algorithm Input
***************

The DBSCAN algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for DBSCAN (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table with the data to be clustered.

       .. note:: The input can be an object of any class derived from ``NumericTable``.

   * - ``weights``
     - Optional input. Pointer to the :math:`n \times 1` numeric table with weights of observations.

       .. note::

         The input can be an object of any class derived from ``NumericTable``
         except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``.

         By default all weights are equal to :math:`1`.

       .. note::

         This parameter is ignored on GPU.

Algorithm Output
****************

The DBSCAN algorithms calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the result of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for DBSCAN (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``assignments``
     - Pointer to the :math:`n \times 1` numeric table with assignments of cluster indices to observations in the input data.

       :term:`Noise observations <noise observation>` have the assignment equal to :math:`-1`.

   * - ``nClusters``
     - Pointer to the :math:`1 \times 1` numeric table with the total number of clusters found by the algorithm.
   * - ``coreIndices``
     - Pointer to the numeric table with :math:`1` column and arbitrary number of rows, containing indices of core observations.
   * - ``coreObservations``
     - Pointer to the numeric table with :math:`p` columns and arbitrary number of rows, containing core observations.

.. note::

    By default, this result is an object of the ``HomogenNumericTable`` class,
    but you can define the result as an object of any class derived from ``NumericTable``
    except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

