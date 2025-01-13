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

.. _cor_cov_distributed:

Distributed Processing
======================

This mode assumes that the data set is split into ``nblocks`` blocks across computation nodes.

Algorithm Parameters
********************

The correlation and variance-covariance matrices algorithm in the distributed processing mode has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Correlation and Variance-Covariance Matrices Algorithm (Distributed Processing)
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

   * - ``outputMatrixType``
     - ``covarianceMatrix``
     - The type of the output matrix. Can be:

       - ``covarianceMatrix`` - variance-covariance matrix
       - ``correlationMatrix`` - correlation matrix

Computation of correlation and variance-covariance matrices follows the general schema described in :ref:`algorithms`:

.. _cor_cov_step_1:

Step 1 - on Local Nodes
***********************
In this step, the correlation and variance-covariance matrices algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Step 1: Algorithm Input for Correlation and Variance-Covariance Matrices Algorithm (Distributed Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the numeric table of size :math:`n_i \times p` that represents the :math:`i`-th data block on the local node.

       While the input for ``defaultDense``, ``singlePassDense``, or ``sumDense`` method can be an object of any class derived
       from ``NumericTable``, the input for ``fastCSR``, ``singlePassCSR``, or ``sumCSR`` method can only be an object of
       the ``CSRNumericTable`` class.

In this step, the correlation and variance-covariance matrices algorithm calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Step 1: Algorithm Output for Correlation and Variance-Covariance Matrices Algorithm (Distributed Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``nObservations``
     - Pointer to the :math:`1 \times 1` numeric table that contains the number of observations processed so far on the local node.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``CSRNumericTable``.
   * - ``crossProduct``
     - Pointer to :math:`p \times p` numeric table with the cross-product matrix computed so far on the local node.

       .. note::

          By default, this table is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
   * - ``sum``
     - Pointer to :math:`1 \times p` numeric table with partial sums computed so far on the local node.

       .. note::

          By default, this table is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

.. _cor_cov_step_2:

Step 2 - on Master Node
***********************
In this step, the correlation and variance-covariance matrices algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Step 2: Algorithm Input for Correlation and Variance-Covariance Matrices Algorithm (Distributed Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``partialResults``
     - A collection that contains results computed in :ref:`Step 1 <cor_cov_step_1>` on local nodes (``nObservations``, ``crossProduct``, and ``sum``).

       .. note::

          The collection can contain objects of any class derived from the ``NumericTable`` class
          except ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

In this step, the correlation and variance-covariance matrices algorithm calculates the results described in the following table.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Step 2: Algorithm Output for for Correlation and Variance-Covariance Matrices Algorithm (Distributed Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``covariance``
     - Use when ``outputMatrixType``=``covarianceMatrix``. Pointer to the numeric table with the :math:`p \times p` variance-covariance matrix.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix`` and ``CSRNumericTable``.
   * - ``correlation``
     - Use when ``outputMatrixType``=``correlationMatrix``. Pointer to the numeric table with the :math:`p \times p` correlation matrix.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix`` and ``CSRNumericTable``.
   * - ``mean``
     - Pointer to the :math:`1 \times p` numeric table with means.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

.. include:: ../../../opt-notice.rst