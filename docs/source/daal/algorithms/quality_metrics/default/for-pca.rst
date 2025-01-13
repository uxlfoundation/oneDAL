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

Quality Metrics for Principal Components Analysis
=================================================

Given the results of the PCA algorithm, data set :math:`E = (e_i)`, :math:`i = \overline{1, p}`
of eigenvalues in decreasing order, full number of principal components :math:`p` and reduced number
of components :math:`p_r \leq p`, the problem is to evaluate the explained variances radio and noise variance.

``QualityMetricsId`` for the PCA algorithm is ``explainedVarianceMetrics``.

Details
*******

The metrics are computed given the input data meets the following requirements:

- At least the largest eigenvalue :math:`e_0` is non-zero. Returns an error otherwise.
- The number of eigenvalues :math:`p` must be equal to the number of features provided.
  Returns an error if :math:`p` is less than the number of features.

The PCA algorithm receives input argument eigenvalues :math:`e_k`, :math:`k = \overline{1, p}`.
It represents the following quality metrics:

- Explained variance ratio
- Noise variance

The library uses the following quality metrics:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Quality Metrics for Principal Components Analysis
   :widths: 10 10
   :header-rows: 1
   :class: longtable

   * - Quality Metric
     - Definition
   * - Explained variance
     - :math:`e_k`, :math:`k = \overline{1, p}`
   * - Explained variance ratios
     - :math:`r_k = \frac {e_k}{\sum _{i = 1}^{p} e_i}`, :math:`k = \overline{1, p}`
   * - Noise variance
     - .. math::
           v_\text{noise} =
           \begin{cases}
              0, & p_r = p;\\
		          \frac{1}{p - p_r} \sum _{i = p_r + 1}^{p} e_i, & p_r < p
           \end{cases}

.. note::
    Quality metrics for PCA are correctly calculated only if the eigenvalues vector obtained from the PCA algorithm has not been reduced.
    That is, the nComponents parameter of the PCA algorithm must be zero or equal to the number of features.
    The formulas rely on a full set of the principal components. If the set is reduced, the result is considered incorrect.

Batch Processing
****************

Algorithm Input
---------------

The Quality Metrics for PCA algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Quality Metrics for Principal Components Analysis (Batch Processing)
   :widths: 10 60
   :header-rows: 1

   * - Input ID
     - Input
   * - ``eigenvalues``
     - :math:`p` eigenvalues (explained variances), numeric table of size :math:`1 \times p`.

       You can define it as an object of any class derived from ``NumericTable`` except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Algorithm Parameters
--------------------

The quality metric algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Quality Metrics for Principal Components Analysis (Batch Processing)
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
   * - ``nComponents``
     - :math:`0`
     - The number of principal components :math:`p_r \leq p` to compute metrics for.
       If it is zero, the algorithm will compute the result for :math:`p`.
   * - ``nFeatures``
     - :math:`0`
     - The number of features in the data set used as input in PCA algorithm.
       If it is zero, the algorithm will compute the result for p.

       .. note:: if :math:`\text{nFeatures} \neq p`, the algorithm will return non-relevant results.

Algorithm Output
----------------

The quality metric for PCA algorithm calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Quality Metrics for Principal Components Analysis (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``explainedVariances``
     - Pointer to the :math:`1 \times p_r` numeric table that contains a reduced eigenvalues array.
   * - ``explainedVariancesRatios``
     - Pointer to the :math:`1 \times p_r` numeric table that contains an array of reduced explained variances ratios.
   * - ``noiseVariance``
     - Pointer to the :math:`1 \times 1` numeric table that contains noise variance.

.. note::
    By default, each numeric table specified by the collection elements is an object of the ``HomogenNumericTable`` class,
    but you can define the result as an object of any class derived from ``NumericTable``, except for ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and CSRNumericTable.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`pca_metrics_dense_batch.cpp <quality_metrics/pca_metrics_dense_batch.cpp>`

