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

.. _dt_regression:

Regression Decision Tree
************************

.. toctree::
   :maxdepth: 1

Regression decision tree is a kind of decision trees described in
Classification and Regression > Decision Tree.

Details
=======

Given:

-  n feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of size p
-  The vector of responses :math:`y = (y_1, \ldots, y_n)` , where :math:`y_i \in R` describes the dependent variable for independent variables :math:`x_i`.

The problem is to build a regression decision tree.

Split Criterion
---------------

The library provides the decision tree regression algorithm based
on the mean-squared error (MSE) [Breiman84]_:

.. math::

  \text{Δ}{I}_{MSE}\left(D, \text{τ}\right)={I}_{MSE}\left(D\right)-\frac{|{D}_{true}|}{|D|}{I}_{MSE}\left({D}_{true}\right)-\frac{|{D}_{false}|}{|D|}{I}_{MSE}\left({D}_{false}\right)

Where

-
   :math:`O(\tau)` is the set of all possible outcomes of test :math:`\tau`

-
   :math:`D_v` is the subset of :math:`D`, for which outcome of :math:`\tau` is :math:`v`, for example, :math:`{D}_{v}=\left\{d\in D\text{|τ}\left(d\right)=v\right\}`.

The test used in the node is selected as :math:`\underset{\tau }{\mathrm{argmax}}\mathrm{\Delta }{I}_{MSE}\left(D, \tau \right)`. For binary decision tree with "true" and "false" branches,

.. math::
   \text{Δ}{I}_{MSE}\left(D, \text{τ}\right)={I}_{MSE}\left(D\right)-\frac{|{D}_{true}|}{|D|}{I}_{MSE}\left({D}_{true}\right)-\frac{|{D}_{false}|}{|D|}{I}_{MSE}\left({D}_{false}\right)

Training Stage
--------------

The regression decision tree follows the algorithmic framework of
decision tree training described in :ref:`decision_tree`.

Prediction Stage
----------------

The regression decision tree follows the algorithmic framework of
decision tree prediction described in :ref:`decision_tree`.

Given the regression decision tree and vectors :math:`x_1, \ldots, x_r`,
the problem is to calculate the responses for those vectors.

Batch Processing
================

Decision tree regression follows the general workflow described
in :ref:`regression_usage_model`.

Training
--------

At the training stage, decision tree regression has the following
parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for Decision Forest Regression (Batch Processing)
   :widths: 10 20 30
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
     - The computation method used by the decision tree regression. The only
       training method supported so far is the default dense method.
   * - ``pruning``
     - ``reducedErrorPruning``
     - Method to perform post-pruning. Available options for the pruning parameter:

       + ``reducedErrorPruning`` - reduced error pruning. Provide dataForPruning and dependentVariablesForPruning inputs, if you use pruning.
       + ``none`` - do not prune.

   * - ``maxTreeDepth``
     - :math:`0`
     - Maximum tree depth. Zero value means unlimited depth. Can be any non-negative number.
   * - ``minObservationsInLeafNodes``
     - :math:`5`
     - Minimum number of observations in the leaf node. Can be any positive number.
   * - ``pruningFraction``
     - :math:`0.2`
     - Fraction of observations from training dataset to be used as
       observations for post-pruning via random sampling. The rest observations
       (with fraction :math:`1 - pruningFraction` to be used to build a decision tree).
       Can be any number in the interval (0, 1). If pruning is not used, all
       observations are used to build the decision tree regardless of this parameter value.
   * - ``engine``
     - `SharedPtr<engines::mt19937::Batch<> >()`
     - Pointer to the random number engine to be used for random sampling for reduced error post-pruning.

Prediction
----------

At the prediction stage, decision tree regression has the following
parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Decision Forest Regression (Batch Processing)
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
     - The computation method used by the decision tree regression. The only
       training method supported so far is the default dense method.

Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`dt_reg_dense_batch.cpp <decision_tree/dt_reg_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`decision_tree_regression.py`
    - :daal4py_example:`decision_tree_regression_traverse.py`
