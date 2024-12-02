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

Multivariate BACON Outlier Detection
====================================

In multivariate outlier detection methods, the observation point is the entire feature vector.

Details
*******

Given a set :math:`X` of :math:`n` feature vectors
:math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of dimension :math:`p`,
the problem is to identify the vectors that do not belong to the underlying distribution using the BACON method (see [Billor2000]_).

In the iterative method, each iteration involves several steps:

#. Identify an initial basic subset of :math:`m > p` feature vectors that can be assumed as not containing outliers.
   The constant :math:`m` is set to :math:`5p`. The library supports two approaches to selecting the initial subset:

   - Based on distances from the medians :math:`||x_i - \text{med}||`, where:

     - `med` is the vector of coordinate-wise medians
     - :math:`||.||` is the vector norm
     - :math:`i = 1, \ldots, n`

   - Based on the Mahalanobis distance :math:`d_i (\text{mean}, S) = \sqrt {(x_i - \text{mean})^T s^{-1} (x_i - \text{mean})}`, where:

     - `mean` and :math:`S` are the mean and the covariance matrix, respectively, of :math:`n` feature vectors
     - :math:`i = 1, \ldots, n`

   Each method chooses :math:`m` feature vectors with the smallest values of distances.

#. Compute the discrepancies using the Mahalanobis distance above, where mean and S are the mean and the covariance matrix, respectively, computed for the feature vectors contained in the basic subset.

#. Set the new basic subset to all feature vectors with the discrepancy less than :math:`c_{npr}\chi_{p, \frac {\alpha}{n}}^2`,
   where:

   - :math:`chi_{p, \alpha}^2` is the :math:`(1 - \alpha)` percentile of the Chi-square distribution with :math:`p` degrees of freedom
   - :math:`c_{npr} = c_{hr} + c_{np}`, where:

     - :math:`r` is the size of the current basic subset
     - :math:`c_{hr} = \max \{0, \frac {h - r}{h + r}\}`, where :math:`h = [\frac{n + p + 1}{2}]` and :math:`[ ]` is the integer part of a number
     - :math:`c_{np} = 1 + \frac{p + 1}{n - p} + \frac{2}{n - 1 - 3p}`

#. Iterate steps 2 and 3 until the size of the basic subset no longer changes.

#. Nominate the feature vectors that are not part of the final basic subset as outliers.

Batch Processing
****************

Algorithm Input
---------------

The multivariate BACON outlier detection algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Multivariate BACON Outlier Detection (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table with the data for outlier detection.

       .. note:: The input can be an object of any class derived from the ``NumericTable`` class.

Algorithm Parameters
--------------------

The multivariate BACON outlier detection algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Multivariate BACON Outlier Detection (Batch Processing)
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
   * - ``initializationMethod``
     - ``baconMedian``
     - The initialization method, can be:

       - ``baconMedian`` - median-based method
       - ``defaultDense`` - Mahalanobis distance-based method
   * - ``alpha``
     - :math:`0.05`
     - One-tailed probability that defines the :math:`(1 - \alpha)` quantile of the :math:`\chi^2` distribution with :math:`p` degrees of freedom.

       Recommended value: :math:`\frac{\alpha}{n}`, where :math:`n` is the number of observations.
   * - ``toleranceToConverge``
     - :math:`0.005`
     - The stopping criterion. The algorithm is terminated if the size of the basic subset is changed by less than the threshold.

Algorithm Output
----------------

The multivariate BACON outlier detection algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Multivariate BACON Outlier Detection (Batch Processing)
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

    - :cpp_example:`out_detect_bacon_dense_batch.cpp <outlier_detection/out_detect_bacon_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`bacon_outlier.py`
