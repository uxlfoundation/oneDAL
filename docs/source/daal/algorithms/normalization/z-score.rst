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

Z-score
=======

Z-score normalization is an algorithm that produces data with each feature (column) having zero mean and unit variance.

Details
*******

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p`, the problem is to compute the matrix :math:`Y = (y_{ij})` of dimension :math:`n \times p` as following:

.. math::

    y_{ij} = \frac {x_{ij} - m_j} {\Delta}

where:

- :math:`m_j` is the mean of :math:`j`-th component of set :math:`(X)_j`, where :math:`j = \overline{1, p}`
- value of :math:`\Delta` depends omn a computation mode

|short_name| provides two modes for computing the result matrix.
You can enable the mode by setting the flag ``doScale`` to a certain position (for details, see `Algorithm Parameters`_).
The mode may include:

- **Centering only.** In this case, :math:`\Delta = 1` and no scaling is performed.
  After normalization, the mean of :math:`j`-th component of result set :math:`(Y)_j` will be zero.

- **Centering and scaling.** In this case, :math:`\Delta = \sigma_j`, where :math:`\sigma_j`
  is the standard deviation of :math:`j`-th component of set :math:`(X)_j`.
  After normalization, the mean of :math:`j`-th component of result set :math:`(Y)_j` will be zero
  and its variance will get a value of one.

.. note::

    Some algorithms require normalization parameters (mean and variance) as an input.
    The implementation of Z-score algorithm in |short_name| does not return these values by default.
    Enable this option by setting the resultsToCompute flag.
    For details, see `Algorithm Parameters`_.

Batch Processing
****************

Algorithm Input
---------------

Z-score normalization algorithm accepts an input as described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Z-score (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the numeric table of size :math:`n \times p`.

       .. note:: This table can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
--------------------

Z-score normalization algorithm has the following parameters.
Some of them are required only for specific values of the computation method parameter ``method``:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.15}|\Y{0.55}|

.. list-table:: Algorithm Parameters for Z-score (Batch Processing)
  :header-rows: 1
  :widths: 10 10 10 60
  :align: left
  :class: longtable

  * - Parameter
    - method
    - Default Value
    - Description
  * - ``algorithmFPType``
    - ``defaultDense`` or ``sumDense``
    - ``float``
    - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
  * - ``method``
    - Not applicable
    - ``defaultDense``
    - Available computation methods:

      defaultDense
        a performance-oriented method. Mean and variance are computed by low order moments algorithm.
        For details, see :ref:`Batch Processing for Moments of Low Order <moments_batch>`.
      sumDense
        a method that uses the basic statistics associated with the numeric table of pre-computed sums.
        Returns an error if pre-computed sums are not defined.
  * - ``moments``
    - ``defaultDense``
    - `SharedPtr<low_order_moments::Batch<algorithmFPType, low_order_moments::defaultDense> >`
    - Pointer to the low order moments algorithm that computes means and standard deviations
      to be used for Z-score normalization with the ``defaultDense`` method.
  * - ``doScale``
    - ``defaultDense`` or ``sumDense``
    - ``true``
    - If true, the algorithm applies both centering and scaling.
      Otherwise, the algorithm provides only centering.
  * - ``resultsToCompute``
    - ``defaultDense`` or ``sumDense``
    - Not applicable
    - *Optional*.

      Pointer to the data collection containing the following key-value pairs for Z-score:

      - ``mean`` - means
      - ``variance`` - variances

      Provide one of these values to request a single characteristic or use bitwise OR to request a combination of them.

Algorithm Output
----------------

Z-score normalization algorithm calculates the result as described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Z-score (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``normalizedData``
     - Pointer to the :math:`n \times p` numeric table that stores the result of normalization.

       .. note::

          By default, the result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``means``
     - *Optional*.

       Pointer to the :math:`1 \times p` numeric table that contains mean values for each feature.

       If the function result is not requested through the ``resultsToCompute`` parameter,
       the numeric table contains a ``NULL`` pointer.
   * - ``variances``
     - *Optional*.

       Pointer to the :math:`1 \times p` numeric table that contains variance values for each feature.

       If the function result is not requested through the ``resultsToCompute`` parameter,
       the numeric table contains a ``NULL`` pointer.     -

.. note::

  By default, each numeric table specified by the collection elements is an object of the ``HomogenNumericTable`` class.
  You can also define the result as an object of any class derived from ``NumericTable``,
  except for ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`zscore_dense_batch.cpp <normalization/zscore_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`normalization_zscore.py`
