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

Distributed Processing
======================

This mode assumes that the data set is split into ``nblocks`` blocks across computation nodes.

Algorithm Parameters
********************

The low order moments algorithm in the distributed processing mode has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Low Order Moments (Distributed Processing)
   :widths: 10 10 60
   :header-rows: 1
   :class: longtable

   * - Parameter
     - Default Valude
     - Description
   * - ``computeStep``
     - Not applicable
     - The parameter required to initialize the algorithm. Can be:

       - ``step1Local`` - the first step, performed on local nodes
       - ``step2Master`` - the second step, performed on a master node

   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Available methods for computation of low order moments:

       defaultDense
           default performance-oriented method

       singlePassDense
           implementation of the single-pass algorithm proposed by D.H.D. West

       sumDense
           implementation of the algorithm in the cases where the basic statistics associated with
           the numeric table are pre-computed sums; returns an error if pre-computed sums are not defined

       fastCSR
           performance-oriented method for CSR numeric tables

       singlePassCSR
           implementation of the single-pass algorithm proposed by D.H.D. West; optimized for CSR numeric tables

       sumCSR
           implementation of the algorithm in the cases where the basic statistics associated with
           the numeric table are pre-computed sums; optimized for CSR numeric tables;
           returns an error if pre-computed sums are not defined

   * - ``estimatesToCompute``
     - ``estimatesAll``
     - Estimates to be computed by the algorithm:

       - ``estimatesAll`` - all supported moments
       - ``estimatesMinMax`` - minimum and maximum
       - ``estimatesMeanVariance`` - mean and variance

Computation of low order moments follows the general schema described in :ref:`algorithms`:

.. _moments_step_1:

Step 1 - on Local Nodes
***********************

In this step, the low order moments algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Low Order Moments (Distributed Processing, Step 1)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the numeric table of size :math:`n_i \times p` that represents the :math:`i`-th data block on the local node.

       While the input for ``defaultDense``, ``singlePassDense``, or ``sumDense`` method can be an object of any class derived
       from ``NumericTable``, the input for ``fastCSR``, ``singlePassCSR``, or ``sumCSR`` method can only be an object of
       the ``CSRNumericTable`` class.

In this step, the low order moments algorithm calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Low Order Moments (Distributed Processing, Step 1)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``nObservations``
     - Pointer to the :math:`1 \times 1` numeric table that contains the number of observations processed so far on the local node.

       By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result
       as an object of any class derived from ``NumericTable`` except ``CSRNumericTable``.

Partial characteristics computed so far on the local node, each in a :math:`1 \times p` numeric table.
By default, each table is an object of the ``HomogenNumericTable`` class, but you can define the tables as objects
of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Partial Characteristics for Low Order Moments (Distributed Processing, Step 1)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``partialMinimum``
     - Partial minimums
   * - ``partialMaximum``
     - Partial maximums
   * - ``partialSum``
     - Partial sums
   * - ``partialSumSquares``
     - Partial sums of squares
   * - ``partialSumSquaresCentered``
     - Partial sums of squared differences from the means

.. _moments_step_2:

Step 2 - on Master Node
***********************

In this step, the low order moments algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Low Order Moments (Distributed Processing, Step 2)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``partialResults``
     - A collection that contains numeric tables with partial results computed in :ref:`Step 1 <moments_step_1>` on local nodes (six numeric tables from each local node).
       These numeric tables can be objects of any class derived from the ``NumericTable`` class except ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

In this step, the low order moments algorithm calculates the results described in the following table.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. note::

    Each result is a pointer to the :math:`1 \times p` numeric table that contains characteristics for each feature in the data set.
    By default, the tables are objects of the ``HomogenNumericTable`` class,
    but you can define each table as an object of any class derived from ``NumericTable``
    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Low Order Moments (Distributed Processing, Step 2)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Characteristic
   * - ``minimum``
     - Minimums
   * - ``maximum``
     - Maximums
   * - ``sum``
     - Sums
   * - ``sumSquares``
     - Sums of squares
   * - ``sumSquaresCentered``
     - Sums of squared differences from the means
   * - ``mean``
     - Estimates for the means
   * - ``secondOrderRawMoment``
     - Estimates for the second order raw moments
   * - ``variance``
     - Estimates for the variances
   * - ``standardDeviation``
     - Estimates for the standard deviations
   * - ``variation``
     - Estimates for the variations

.. include:: ../../../opt-notice.rst
