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

Multivariate Outlier Detection
==============================

In multivariate outlier detection methods, the observation point is the entire feature vector.

Details
*******

Given a set :math:`X` of :math:`n` feature vectors
:math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of dimension :math:`p`,
the problem is to identify the vectors that do not belong to the underlying distribution
(see [Ben2005]_ for exact definitions of an outlier).

The multivariate outlier detection method takes into account dependencies between features.
This method can be parametric, assumes a known underlying distribution for the data set, and defines an outlier region
such that if an observation belongs to the region, it is marked as an outlier.
Definition of the outlier region is connected to the assumed underlying data distribution.

The following is an example of an outlier region for multivariate outlier detection:

.. math::

    \text{Outlier}(\alpha_n, M_n, \Sigma_n) = \{x: \sqrt{(x - M_n) \sum _{n}{-1} (x - M_n)} > g(n, \alpha_n) \}

where :math:`M_n` and \Sigma_n are (robust) estimates of the vector of means and variance-covariance matrix computed for a given data set,
:math:`\alpha_n` is the confidence coefficient, and :math:`g(n, \alpha_n)` defines the limit of the region.

Batch Processing
****************

Algorithm Input
---------------

The multivariate outlier detection algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Multivariate Outlier Detection (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table with the data for outlier detection.
       The input can be an object of any class derived from the ``NumericTable`` class.
   * - ``location``
     - Pointer to the :math:`1 \times p` numeric table with the vector of means.
       The input can be an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.
   * - ``scatter``
     - Pointer to the :math:`p \times p` numeric table that contains the variance-covariance matrix.
       The input can be an object of any class derived from ``NumericTable`` except ``PackedTriangularMatrix``.
   * - ``threshold``
     - Pointer to the :math:`1 \times 1` numeric table with the non-negative number that defines the outlier region.
       The input can be an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

If you do not provide at least one of the ``location``, ``scatter``, ``threshold`` inputs,
the library will initialize all of them with the following default values:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Default Values for Algorithm Input of Multivariate Outlier Detection (Batch Processing)
    :widths: 10 20
    :class: longtable

    * - ``location``
      - A set of :math:`0.0`
    * - ``scatter``
      - A numeric table with diagonal elements equal to :math:`1.0` and non-diagonal elements equal to :math:`0.0`
    * - ``threshold``
      - :math:`3.0`

Algorithm Parameters
--------------------

The multivariate outlier detection algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Multivariate Outlier Detection (Batch Processing)
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
     - Performance-oriented computation method.

Algorithm Output
----------------

The multivariate outlier detection algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Multivariate Outlier Detection (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Result ID
     - Result
   * - ``weights``
     - Pointer to the :math:`n \times 1` numeric table of zeros and ones.
       Zero in the :math:`i`-th position indicates that the :math:`i`-th feature vector is an outlier.

       .. note::
          By default, the result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except the ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`out_detect_mult_dense_batch.cpp <outlier_detection/out_detect_mult_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`multivariate_outlier.py`

Performance Considerations
**************************

To get the best overall performance of multivariate outlier detection:

- If input data is homogeneous, provide input data and store results in homogeneous numeric tables
  of the same type as specified in the ``algorithmFPType`` class template parameter.
- If input data is non-homogeneous, use :ref:`AOS layout <aos_layout>` rather than SOA layout.
- For the default outlier detection method (``defaultDense``),
  you can benefit from splitting the input data set into blocks for parallel processing.

.. include:: ../../../opt-notice.rst