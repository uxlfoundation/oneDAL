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

Univariate Outlier Detection
============================

A univariate outlier is an occurrence of an abnormal value within a single observation point.

Details
*******

Given a set :math:`X` of :math:`n` feature vectors
:math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of dimension :math:`p`,
the problem is to identify the vectors that do not belong to the underlying distribution
(see [Ben2005]_ for exact definitions of an outlier).

The algorithm for univariate outlier detection considers each feature independently.
The univariate outlier detection method can be parametric, assumes a known underlying distribution for the data set,
and defines an outlier region such that if an observation belongs to the region, it is marked as an outlier.
Definition of the outlier region is connected to the assumed underlying data distribution.

The following is an example of an outlier region for the univariate outlier detection:

.. math::

    \text{Outlier}(\alpha_n, m_n, \sigma_n) = \{x: \frac {|x - m_n|}{\sigma_n} > g(n, \alpha_n) \}


where :math:`m_n` and :math:`\sigma_n` are (robust) estimates of the mean and standard deviation
computed for a given data set, :math:`\alpha_n` is the confidence coefficient,
and :math:`g(n, \alpha_n)` defines the limits of the region and should be adjusted to the number of observations.

Batch Processing
****************

Algorithm Input
---------------

The univariate outlier detection algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Univariate Outlier Detection (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table with the data for outlier detection.

       .. note:: The input can be an object of any class derived from the ``NumericTable`` class.
   * - ``location``
     - Pointer to the :math:`1 \times p` numeric table with the vector of means.

       .. note:: The input can be an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.
   * - ``scatter``
     - Pointer to the :math:`1 \times p` numeric table with the vector of standard deviations.

       .. note:: The input can be an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.
   * - ``threshold``
     - Pointer to the :math:`1 \times p` numeric table with non-negative numbers that define the outlier region.

       .. note:: The input can be an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

If you do not provide at least one of the ``location``, ``scatter``, ``threshold`` inputs,
the library will initialize all of them with the following default values:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Default Values for Algorithm Input of Univariate Outlier Detection (Batch Processing)
    :widths: 10 20
    :class: longtable

    * - ``location``
      - A set of :math:`0.0`
    * - ``scatter``
      - A set of :math:`1.0`
    * - ``threshold``
      - A set of :math:`3.0`

Algorithm Parameters
--------------------

The univariate outlier detection algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Univariate Outlier Detection (Batch Processing)
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

Algorithm Output
----------------

The univariate outlier detection algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Univariate Outlier Detection (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``weights``
     - Pointer to the :math:`n \times p` numeric table of zeros and ones.
       Zero in the position :math:`(i, j)` indicates an outlier in the :math:`i`-th observation of the :math:`j`-th feature.

       .. note::

          By default, the result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``СSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`out_detect_uni_dense_batch.cpp <outlier_detection/out_detect_uni_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`univariate_outlier.py`
