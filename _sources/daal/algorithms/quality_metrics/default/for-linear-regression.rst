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

Quality Metrics for Linear Regression
=====================================

Given a data set :math:`X = (x_i)` that contains vectors of input variables :math:`x_i = (x_{i1}, \ldots, x_{ip})`,
respective responses :math:`z_i = (z_{i1}, \ldots, z_{ik})` computed at the prediction stage of the linear regression model
defined by its coefficients :math:`\beta_{ht}`, :math:`h = 1, \ldots, k`, :math:`t = 1, \ldots, p`,
and expected responses :math:`y_i = (y_{i1}, \ldots, y_{ik})`, :math:`i = 1, \ldots, n`,
the problem is to evaluate the linear regression model by computing the root mean square error,
variance-covariance matrix of beta coefficients, various statistics functions, and so on.
See :ref:`linear_regression` for additional details and notations.

For linear regressions, the library computes statistics listed in tables below
for testing insignificance of beta coefficients and one of the following values of ``QualityMetricsId``:

- ``singleBeta`` for :ref:`a single coefficient <qm_lin_reg_single_beta>`
- ``groupOfBetas`` for :ref:`a group of coefficients <qm_lin_reg_group_of_betas>`

For more details, see [Hastie2009]_.

Details
*******

The statistics are computed given the following assumptions about the data distribution:

- Responses :math:`y_{ij}`, :math:`i = 1, \ldots, n`, are independent and have a constant variance :math:`\sigma_j^2`, :math:`j = 1, \ldots, k`
- Conditional expectation of responses :math:`y_{.j}`, :math:`j = 1, \ldots, k`, is linear in input variables :math:`x_{.} = (x_{.1}, \ldots , x_{.p})`
- Deviations of :math:`y_{ij}`, :math:`i = 1, \ldots, n`, around the mean of expected responses :math:`\text{ERM}_j`, :math:`j = 1, \ldots, k`, are additive and Gaussian.

.. _qm_lin_reg_single_beta:

Testing Insignificance of a Single Beta
---------------------------------------

The library uses the following quality metrics:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Quality Metrics for Testing Insignificance of a Single Beta
   :widths: 10 10
   :header-rows: 1
   :class: longtable

   * - Quality Metric
     - Definition
   * - Root Mean Square (RMS) Error
     - :math:`\sqrt{\frac{1}{n} \sum _{i=1}^{n} (y_{ij} - x_{ij})^2}`, :math:`j = 1, \ldots, k`
   * - Vector of variances :math:`\sigma^2 = (\sigma_1^2, \ldots, \sigma_k^2)`
     - :math:`\sigma_j^2 = \frac {1}{n - p - 1} \sum _{i=1}^{n} (y_{ij} - x_{ij})^2`, :math:`j = 1, \ldots, k`
   * - A set of variance-covariance matrices :math:`C = C_1, \ldots, C_k` for vectors of betas :math:`\beta_{jt}`, :math:`j = 1, \ldots, k`
     - :math:`C_j = {(X^T X)}^{-1} \sigma_j^2`, :math:`j = 1, \ldots, k`
   * - Z-score statistics used in testing of insignificance of a single coefficient :math:`\beta_{jt}`
     - :math:`\text{zscore}_{jt} = \frac {\beta_{jt}}{\sigma_j \sqrt{v_t}}`, :math:`j = 1, \ldots, k`,
       :math:`\sigma_j` is the :math:`j`-th element of the vector of variance :math:`\sigma^2` and
       :math:`ν_t` is the :math:`t`-th diagonal element of the matrix :math:`{(X^T X)}^{-1}`
   * - Confidence interval for :math:`\beta_{jt}`
     - :math:`(\beta_{jt} - \text{pc}_{1-\alpha} \sqrt{v_t}, \beta_{jt} + \text{pc}_{1-\alpha} \sqrt{v_t})`, :math:`j = 1, \ldots, k`,
       :math:`\text{pc}_{1-\alpha}` is the :math:`(1-\alpha)` percentile of the Gaussian distribution,
       :math:`\sigma_j` is the :math:`j`-th element of the vector of variance :math:`\sigma^2`,
       :math:`ν_t` is the :math:`t`-th diagonal element of the matrix :math:`{(X^T X)}^{-1}`

.. _qm_lin_reg_group_of_betas:

Testing Insignificance of a Group of Betas
------------------------------------------

The library uses the following quality metrics:

.. tabularcolumns::  |\Y{0.3}|\Y{0.7}|

.. list-table:: Quality Metrics for Testing Insignificance of a Group of Betas
   :widths: 10 10
   :header-rows: 1
   :class: longtable

   * - Quality Metric
     - Definition
   * - Mean of expected responses, :math:`\text{ERM} = (\text{ERM}_1, \ldots, \text{ERM}_k)`
     - :math:`\text{ERM}_j = \frac {1}{n} \sum _{i=1}^{n} y_{ij}`, :math:`j = 1, \ldots, k`
   * - Variance of expected responses, :math:`\text{ERV} = (\text{ERV}_1, \ldots, \text{ERV}_k)`
     - :math:`\text{ERV}_j = \frac {1}{n - 1} \sum _{i=1}^{n} (y_{ij} - \text{ERM}_j)^2`, :math:`j = 1, \ldots, k`
   * - Regression Sum of Squares :math:`\text{RegSS} = (\text{RegSS}_1, \ldots, \text{RegSS}_k)`
     - :math:`\text{RegSS}_j = \frac {1}{n} \sum _{i=1}^{n} (z_{ij} - \text{ERM}_j)^2`, :math:`j = 1, \ldots, k`
   * - Sum of Squares of Residuals :math:`\text{ResSS} = (\text{ResSS}_1, \ldots, \text{ResSS}_k)`
     - :math:`\text{ResSS}_j = \sum _{i=1}^{n} (y_{ij} - z_{ij})^2`, :math:`j = 1, \ldots, k`
   * - Total Sum of Squares :math:`\text{TSS} = (\text{TSS}_1, \ldots, \text{TSS}_k)`
     - :math:`\text{TTS}_j = \sum _{i=1}^{n} (y_{ij} - \text{ERM}_j)^2`, :math:`j = 1, \ldots, k`
   * - Determination Coefficient :math:`R^2 = (R_1^2, \ldots, R_k^2)`
     - :math:`R^2_j = \frac {\text{RegSS}_j}{\text{TTS}_j }`, :math:`j = 1, \ldots, k`
   * - F-statistics used in testing insignificance of a group of betas :math:`F = (F_1, \ldots, F_k)`
     - :math:`F_j = \frac {(\text{ResSS}_{0j} - \text{ResSS}_j)/(p - p_0)}
       {{\text{ResSS}_j}/(n - p - 1)}`, :math:`j = 1, \ldots, k`,
       where :math:`\text{ResSS}_j` are computed for a model with :math:`p + 1` betas and
       :math:`\text{ResSS}_{0j}` are computed for a reduced model with :math:`p_0 + 1` betas (:math:`p - p_0` betas are set to zero)

Batch Processing
****************

- :ref:`qm_lin_reg_single_beta_batch`
- :ref:`qm_lin_reg_group_of_betas_batch`

.. _qm_lin_reg_single_beta_batch:

Testing Insignificance of a Single Beta
---------------------------------------

.. rubric:: Algorithm Input

The quality metric algorithm for linear regression accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Testing Insignificance of a Single Beta in Linear Regression (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``expectedResponses``
     - Pointer to the :math:`n \times k` numeric table with responses (:math:`k` dependent variables) used for training the linear regression model.

       This table can be an object of any class derived from ``NumericTable``.
   * - ``model``
     - Pointer to the model computed at the training stage of the linear regression algorithm.

       The model can only be an object of the ``linear_regression::Model`` class.
   * - ``predictedResponses``
     - Pointer to the :math:`n \times k` numeric table with responses (:math:`k` dependent variables) computed at the prediction stage of the linear regression algorithm.

       This table can be an object of any class derived from ``NumericTable``.

.. rubric::  Algorithm Parameters

The quality metric algorithm for linear regression has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Testing Insignificance of a Single Beta in Linear Regression (Batch Processing)
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
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``alpha``
     - :math:`0.05`
     - Significance level used in the computation of confidence intervals for coefficients of the linear regression model.
   * - ``accuracyThreshold``
     - :math:`0.001`
     - Values below this threshold are considered equal to it.


.. rubric::  Algorithm Output

The quality metric algorithm for linear regression calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Testing Insignificance of a Single Beta in Linear Regression (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``rms``
     - Pointer to the :math:`1 \times k` numeric table that contains root mean square errors computed for each response (dependent variable)

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable``, except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

   * - ``variance``
     - Pointer to the :math:`1 \times k` numeric table that contains variances :math:`\sigma^2_j`, :math:`j = 1, \ldots, k` computed for each response (dependent variable).

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable``, except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

   * - ``betaCovariances``
     - Pointer to the DataCollection object that contains :math:`k` numeric tables, each with the :math:`m \times m` variance-covariance matrix for betas of the j-th response (dependent variable), where m is the number of betas in the model (m is equal to p when interceptFlag is set to false at the training stage of the linear regression algorithm; otherwise, m is equal to p + 1 ).

       The collection can contain objects of any class derived from ``NumericTable``.
   * - ``zScore``
     - Pointer to the :math:`k \times m` numeric table that contains the Z-score statistics used in the testing of insignificance of individual linear regression coefficients,
       where :math:`m` is the number of betas in the model (:math:`m` is equal to :math:`p` when ``interceptFlag`` is set to ``false`` at the training stage
       of the linear regression algorithm; otherwise, :math:`m` is equal to :math:`p + 1`).

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable``, except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

   * - ``confidenceIntervals``
     - Pointer to the :math:`k \times 2 \times m` numeric table that contains limits of the confidence intervals for linear regression coefficients:

       - :math:`\text{confidenceIntervals}[t][2*j]` is the left limit of the confidence interval
         computed for the :math:`j`-th beta of the :math:`t`-th response (dependent variable)
       - :math:`\text{confidenceIntervals}[t][2*j+1]` is the right limit of the confidence interval
         computed for the :math:`j`-th beta of the :math:`t`-th response (dependent variable),

       where :math:`m` is the number of betas in the model (:math:`m` is equal to :math:`p` when ``interceptFlag`` is set to ``false`` at the training stage
       of the linear regression algorithm; otherwise, :math:`m` is equal to :math:`p + 1`).

       .. note::
           By default, this result is an object of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
           derived from ``NumericTable``, except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

   * - ``inverseOfXtX``
     - Pointer to the :math:`m \times m` numeric table that contains the :math:`{(X^TX)}^{-1}` matrix,
       where :math:`m` is the number of betas in the model (:math:`m` is equal to :math:`p` when ``interceptFlag`` is set to ``false`` at the training stage
       of the linear regression algorithm; otherwise, :math:`m` is equal to :math:`p + 1`).


.. _qm_lin_reg_group_of_betas_batch:

Testing Insignificance of a Group of Betas
------------------------------------------

.. rubric:: Algorithm Input

The quality metric algorithm for linear regression accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Testing Insignificance of a Group of Betas in Linear Regression (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Input ID
     - Input
   * - ``expectedResponses``
     - Pointer to the :math:`n \times k` numeric table with responses (:math:`k` dependent variables) used for training the linear regression model.

       This table can be an object of any class derived from ``NumericTable``.
   * - ``predictedResponses``
     - Pointer to the :math:`n \times k` numeric table with responses (:math:`k` dependent variables) computed at the prediction stage of the linear regression algorithm.

       This table can be an object of any class derived from ``NumericTable``.
   * - ``predictedReducedModelResponses``
     - Pointer to the :math:`n \times k` numeric table with responses (:math:`k` dependent variables) computed at the prediction stage of the linear regression algorithm
       using the reduced linear regression model, where :math:`p - p_0` out of :math:`p` beta coefficients are set to zero.

       This table can be an object of any class derived from ``NumericTable``.

.. rubric:: Algorithm Parameters

The quality metric algorithm for linear regression has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Testing Insignificance of a Group of Betas in Linear Regression (Batch Processing)
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
   * - ``method``
     - ``defaultDense``
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``numBeta``
     - :math:`0`
     - Number of beta coefficients used for prediction.
   * - ``numBetaReducedModel``
     - :math:`0`
     - Number of beta coefficients (:math:`p_0`) used for prediction with the reduced linear regression model,
       where :math:`p - p_0` out of :math:`p` beta coefficients are set to zero.

.. rubric:: Algorithm Output

The quality metric algorithm for linear regression calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.
For more details, see :ref:`algorithms`.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Testing Insignificance of a Group of Betas in Linear Regression (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Result ID
     - Result
   * - ``expectedMeans``
     - Pointer to the :math:`1 \times k` numeric table that contains the mean of expected responses computed for each dependent variable.
   * - ``expectedVariance``
     - Pointer to the :math:`1 \times k` numeric table that contains the variance of expected responses computed for each dependent variable.
   * - ``regSS``
     - Pointer to the :math:`1 \times k` numeric table that contains the regression sum of squares computed for each dependent variable.
   * - ``resSS``
     - Pointer to the :math:`1 \times k` numeric table that contains the sum of squares of residuals computed for each dependent variable.
   * - ``tSS``
     - Pointer to the :math:`1 \times k` numeric table that contains the total sum of squares computed for each dependent variable.
   * - ``determinationCoeff``
     - Pointer to the :math:`1 \times k` numeric table that contains the determination coefficient computed for each dependent variable.
   * - ``fStatistics``
     - Pointer to the :math:`1 \times k` numeric table that contains the F-statistics computed for each dependent variable.

.. note::
    By default, these results are objects of the ``HomogenNumericTable`` class, but you can define the result as an object of any class
    derived from ``NumericTable``, except for ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.


Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`lin_reg_metrics_dense_batch.cpp <quality_metrics/lin_reg_metrics_dense_batch.cpp>`

