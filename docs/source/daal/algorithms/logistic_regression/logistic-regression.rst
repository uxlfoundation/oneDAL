.. Copyright 2019 Intel Corporation
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

.. _logistic_regression:

Logistic Regression
===================

.. toctree::
   :glob:
   :maxdepth: 4


Logistic regression is a method for modeling the relationships
between one or more explanatory variables and a categorical variable
by expressing the posterior statistical distribution of the
categorical variable via linear functions on observed data. If the
categorical variable is binary, taking only two values, "0" and "1",
the logistic regression is simple, otherwise, it is multinomial.


Details
*******

Given n feature vectors of n p-dimensional feature vectors a vector
of class labels :math:`y = (y_1,\ldots{},y_n )`, where :math:`y_i \in \{0, 1, \ldots, K-1\}` and :math:`K` is the number of classes, describes the
class to which the feature vector :math:`x_i` belongs, the problem is to
train a logistic regression model.

The logistic regression model is the set of vectors :math:`\beta =\left\{ {\beta }_{0}=\left({\beta }_{00}\dots {\beta }_{0p}\right), {..\beta }_{K-1}=\left({\beta }_{K-10}\dots {\beta }_{K-1p}\right)\right\}` that gives the posterior probability

.. math::
   {P\left\{y=k|x\right\}= p}_{k}\left(x, \beta \right)=\mathrm{ }\frac{{e}^{{f}_{k}\left(x, \beta \right)}}{\sum _{i=0}^{K-1}{e}^{{f}_{i}\left(x, \beta \right)}}, \text{where} {f}_{k}\left(x, \beta \right)= {\beta }_{k0}+ \sum _{j=1}^{p}{\beta }_{kj}*{x}_{j}

for a given feature vector :math:`x = (x_1, \ldots, x_p)` and class
label :math:`y \in \{0, 1, \ldots, K - 1\}` for each :math:`k = 0, \ldots, K-1`. See [Hastie2009]_.

If the categorical variable is binary, the model is defined as a single vector :math:`{\beta }_{0}=\left({\beta }_{00}\dots {\beta }_{0p}\right)` that determines the posterior probability

.. math::

		 P\left\{y=1|x\right\}= \mathrm{\sigma }\left(x,\mathrm{ }\mathrm{\beta }\right)=\frac{1}{1+ {e}^{-f\left(x, \beta \right)}}

		 P\left\{y=0|x\right\}=1- P\left\{y=1|x\right\}


Training Stage
--------------

Training procedure is an iterative algorithm which minimizes
objective function

.. math::
		 L\left(\beta \right)=-\frac{1}{n}\sum _{i=1}^{n}\mathrm{log}{p}_{{y}_{i}}\left({x}_{i},\mathrm{ }\mathrm{\beta }\right)+\mathrm{ }{\lambda }_{1}\sum _{k=0}^{K-1}{\|{\beta }_{k}\|}_{L1}+ {\lambda }_{2}\sum _{k=0}^{K-1}{\|{\beta }_{k}\|}_{L2}

where the first term is the negative log-likelihood of conditional
:math:`Y` given :math:`X`, and the latter terms are regularization ones that
penalize the complexity of the model (large :math:`\beta` values), :math:`\lambda_1` and :math:`\lambda_2` are non-negative regularization
parameters applied to L1 and L2 norm of vectors in :math:`\beta`.

For more details, see [Hastie2009]_, [Bishop2006]_.

For the objective function minimization the library supports the
iterative algorithms defined by the interface of
|namespace|::algorithms::iterative_solver. See :ref:`iterative_solver`.

Prediction Stage
-----------------

Given logistic regression model and vectors :math:`x_1, \ldots, x_r`, the problem is to calculate the responses for those
vectors, and their probabilities and logarithms of probabilities
if required. The computation is based on formula (1) in
multinomial case and on formula (2) in binary case.

Usage of Training Alternative
*****************************

To build a Logistic Regression model using methods of the Model Builder class of Logistic Regression,
complete the following steps:

- Create a Logistic Regression model builder using a constructor with the required number of responses and features.
- Use the ``setBeta`` method to add the set of pre-calculated coefficients to the model.
  Specify random access iterators to the first and the last element of the set of coefficients [ISO/IEC 14882:2011 §24.2.7]_.

  .. note::

    If your set of coefficients does not contain an intercept, interceptFlag is automatically set to ``False``, and to ``True``, otherwise.

- Use the ``getModel`` method to get the trained Logistic Regression model.
- Use the ``getStatus`` method to check the status of the model building process.
  If ``DAAL_NOTHROW_EXCEPTIONS`` macros is defined, the status report contains the list of errors
  that describe the problems API encountered (in case of API runtime failure).

.. note::

  If after calling the ``getModel`` method you use the ``setBeta`` method to update coefficients,
  the initial model will be automatically updated with the new :math:`\beta` coefficients.

Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

    - :cpp_example:`log_reg_model_builder.cpp <logistic_regression/log_reg_model_builder.cpp>`

  .. tab:: Python*

    - :daal4py_example:`log_reg_model_builder.py <https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/examples/mb/log_reg_model_builder.py>`

Batch Processing
****************

Logistic regression algorithm follows the general workflow described
in :ref:`classification_usage_model`.

Training
--------

For a description of the input and output, refer to :ref:`classification_usage_model`.

In addition to the parameters of classifier described in :ref:`classification_usage_model`,
the logistic regression batch training algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Training Parameters for Logistic Regression (Batch Processing)
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
     - The computation method used by the logistic regression. The only
       training method supported so far is the default dense method.
   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.
   * - ``interceptFlag``
     - ``True``
     - A flag that indicates a need to compute :math:`\theta_j`
   * - ``penaltyL1``
     - :math:`0`
     - L1 regularization coefficient

       .. note:: L1 regularization is not supported on GPU.

   * - ``penaltyL2``
     - :math:`0`
     - L2 regularization coefficient
   * - ``optimizationSolver``
     - :ref:`SGD solver <sgd_solver>`
     - All iterative solvers are available as optimization procedures to use at the training stage:

       -  :ref:`SGD (Stochastic Gradient Descent Algorithm) <sgd_solver>`
       -  :ref:`ADAGRAD (Adaptive Subgradient Method) <adagrad_solver>`
       -  :ref:`LBFGS (Limited-Memory Broyden-Fletcher-Goldfarb-Shanno Algorithm) <lbfgs_solver>`
       -  :ref:`SAGA (Stochastic Average Gradient Accelerated Method) <saga_solver>`

Prediction
----------

For a description of the input, refer to :ref:`classification_usage_model`.

At the prediction stage logistic regression batch algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Logistic Regression (Batch Processing)
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate
       computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - The computation method used by logistic regression. The only prediction
       method supported so far is the default dense method.
   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.
   * - ``resultsToCompute``
     - ``computeClassesLabels``
     - The 64-bit integer flag that specifies which extra characteristics of
       the logistic regression to compute.

       Provide one of the following values to request a single characteristic
       or use bitwise OR to request a combination of the characteristics:

       - ``computeClassesLabels`` for `prediction`
       - ``computeClassesProbabilities`` for `probabilities`
       - ``computeClassesLogProbabilities`` for `logProbabilities`

Output
------

In addition to classifier output, logistic regression prediction calculates the result described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Prediction Output for Logistic Regression (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``probabilities``
     - A numeric table of size :math:`n \times nClasses` containing probabilities of classes computed when ``computeClassesProbabilities`` option is enabled.

   * - ``logProbabilities``
     - A numeric table of size :math:`n \times nClasses` containing logarithms of classes' probabilities computed when ``computeClassesLogProbabilities`` option is enabled.



.. note::

   Note that:

   - If `resultsToCompute` does not contain `computeClassesLabels`, the `prediction` table is `NULL`.

   - If `resultsToCompute` does not contain `computeClassesProbabilities`, the `probabilities` table is `NULL`.

   - If `resultsToCompute` does not contain `computeClassesLogProbabilities`, the `logProbabilities` table is `NULL`.

   -  By default, each numeric table of this result is an object of
      the ``HomogenNumericTable`` class, but you can define the result as
      an object of any class derived from ``NumericTable`` except for
      ``PackedSymmetricMatrix`` and ``PackedTriangularMatrix``.

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    -  :cpp_example:`log_reg_dense_batch.cpp <logistic_regression/log_reg_dense_batch.cpp>`
    -  :cpp_example:`log_reg_binary_dense_batch.cpp <logistic_regression/log_reg_binary_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`log_reg_dense.py`
    - :daal4py_example:`log_reg_binary_dense.py`
