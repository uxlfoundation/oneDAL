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

Algorithm Input
---------------
The PCA algorithm accepts the input described below. Pass the
``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Principal Component Analysis (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Use when the input data is a normalized or non-normalized data set.
       Pointer to the :math:`n \times p` numeric table that contains the input data set.

       .. note:: This input can be an object of any class derived from ``NumericTable``.
   * - ``correlation``
     - Use when the input data is a correlation matrix. Pointer to the :math:`p \times p`
       numeric table that contains the correlation matrix.

       .. note::
          This input can be an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``.

Algorithm Parameters
--------------------

The PCA algorithm has the following parameters, depending on the
computation method parameter method:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.3}|\Y{0.4}|

.. list-table:: Algorithm Parameters for Principal Component Analysis (Batch Processing)
   :widths: 10 10 15 25
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - method
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``defaultDense`` or ``svdDense``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be ``float`` or ``double``.
   * - ``method``
     - Not applicable
     - ``defaultDense``
     - Available methods for PCA computation:

       For CPU:

       - ``defaultDense`` - the correlation method
       - ``svdDense`` - the SVD method

       For GPU:

       - ``defaultDense`` - the correlation method


   * - ``covariance``
     - ``defaultDense``
     - `SharedPtr<covariance::Batch<algorithmFPType, covariance::defaultDense> >`
     - The correlation and variance-covariance matrices algorithm to be used
       for PCA computations with the correlation method.
   * - ``normalization``
     - ``svdDense``
     - `SharedPtr<normalization::zscore::Batch<algorithmFPType, normalization::zscore::defaultDense>>`
     - The data normalization algorithm to be used for PCA computations with
       the SVD method.
   * - ``nComponents``
     - ``defaultDense``, ``svdDense``
     - :math:`0`
     - The number of principal components :math:`p_r`. If it is zero, the algorithm
       will compute the result for :math:`p_r = p`.
   * - ``isDeterministic``
     - ``defaultDense``, ``svdDense``
     - ``false``
     - If true, the algorithm applies the "sign flip" technique to the results.
   * - ``resultsToCompute``
     - ``defaultDense``, ``svdDense``
     - ``none``
     - The 64-bit integer flag that specifies which optional result to compute.

       Provide one of the following values to request a single characteristic
       or use bitwise OR to request a combination of the characteristics:

       -  ``mean``
       -  ``variance``
       -  ``eigenvalue``

Algorithm Output
----------------

The PCA algorithm calculates the results described below. Pass the
``Result ID`` as a parameter to the methods that access the results of
your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Principal Component Analysis (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``eigenvalues``
     - Pointer to the :math:`1 \times p_r` numeric table that contains eigenvalues
       in the descending order.

       .. note::
          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
   * - ``eigenvectors``
     - Pointer to the :math:`p_r \times p` numeric table that contains eigenvectors
       in the row-major order.

       .. note::
          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

   * - ``means``
     - Pointer to the :math:`1 \times p_r` numeric table that contains mean values
       for each feature.
       Optional.
       If correlation is provided then the vector is filed with zeroes.
   * - ``variances``
     - Pointer to the :math:`1 \times p_r` numeric table that contains mean values
       for each feature.
       Optional.
       If correlation is provided then the vector is filed with zeroes.
   * - ``dataForTransform``
     - Pointer to key value data collection containing the aggregated data for
       normalization and whitening with the following key value pairs:

       -  mean - mean
       -  variance - variance
       -  eigenvalue - eigenvalue

       If ``resultsToCompute`` does not contain mean, the dataForTransform means
       table is NULL. If ``resultsToCompute`` does not contain variances, the
       ``dataForTransform`` variances table is NULL. If ``resultsToCompute`` does not
       contain eigenvalues, the ``dataForTransform`` eigenvalues table is NULL.

Please note the following:

.. note::

   -  If the function result is not requested through the
      ``resultsToCompute`` parameter, the respective element of the result
      contains a NULL pointer.
   -  By default, each numeric table specified by the collection
      elements is an object of the ``HomogenNumericTable`` class, but you
      can define the result as an object of any class derived from
      ``NumericTable``, except for ``PackedSymmetricMatrix``,
      ``PackedTriangularMatrix``, and ``CSRNumericTable``.
   -  For the ``svdDense`` method :math:`n` should not be less than :math:`p`. If :math:`n > p`,
      svdDense returns an error.
