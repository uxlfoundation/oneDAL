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

Algorithm Input
***************

The correlation and variance-covariance matrices algorithm accepts
the input described below. Pass the ``Input ID`` as a parameter to the
methods that provide input for your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Correlation and Variance-Covariance Matrices Algorithm (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table for which the variance-covariance or
       correlation matrix :math:`C` is computed. While the input for ``defaultDense``,
       ``singlePassDense``, or ``sumDense`` method can be an object of any class
       derived from ``NumericTable``, the input for ``fastCSR``, ``singlePassCSR``, or
       ``sumCSR`` method can only be an object of the ``CSRNumericTable`` class.

Algorithm Parameters
********************

The correlation and variance-covariance matrices algorithm has the
following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Algorithm Parameters for Correlation and Variance-Covariance Matrices Algorithm (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 20 30
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Available methods for computation of correlation and variance-covariance matrices:

       For CPU:

       + ``defaultDense`` - default performance-oriented method
       + ``singlePassDense`` - implementation of the single-pass algorithm proposed by D.H.D. West
       + ``sumDense`` - implementation of the algorithm in the cases where the
         basic statistics associated with the numeric table are pre-computed
         sums; returns an error if pre-computed sums are not defined
       + ``fastCSR`` - performance-oriented method for CSR numeric tables
       + ``singlePassCSR`` - implementation of the single-pass algorithm proposed by D.H.D. West; optimized for CSR numeric tables
       + ``sumCSR`` - implementation of the algorithm in the cases where the basic
         statistics associated with the numeric table are pre-computed sums;
         optimized for CSR numeric tables; returns an error if pre-computed
         sums are not defined

       For GPU:

       + ``defaultDense`` - default performance-oriented method

   * - ``outputMatrixType``
     - ``covarianceMatrix``
     - The type of the output matrix. Can be:

       + ``covarianceMatrix`` - variance-covariance matrix
       + ``correlationMatrix`` - correlation matrix

Algorithm Output
****************

The correlation and variance-covariance matrices algorithm calculates
the result described below. Pass the ``Result ID`` as a parameter to the
methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Correlation and Variance-Covariance Matrices Algorithm (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60
   :class: longtable

   * - Result ID
     - Result
   * - ``covariance``
     - Use when outputMatrixType=covarianceMatrix. Pointer to the numeric table
       with the :math:`p \times p` variance-covariance matrix.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix`` and ``CSRNumericTable``.
   * - ``correlation``
     - Use when outputMatrixType=correlationMatrix. Pointer to the numeric
       table with the :math:`p \times p` correlation matrix.

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