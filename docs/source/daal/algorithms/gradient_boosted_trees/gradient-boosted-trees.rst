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

.. _gb_trees:

Gradient Boosted Trees
======================

Details
*******

Given n feature vectors  :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}` of :math:`n`
:math:`p`-dimensional feature vectors and :math:`n` responses :math:`Y = \{y_1, \ldots, y_n \}`, the problem is to build a gradient boosted trees
classification or regression model.

The tree ensemble model uses M additive functions to predict the output :math:`\hat{y_i}=f(x)={\sum }_{k=1}^{M}{f}_{k}({x}_{i}), {f}_{k}\in F`
where :math:`F=\mathrm{ }\left\{f\left(x\right)={w}_{q\left(x\right)}\mathrm{ },\mathrm{ }q:{R}^{p}\to T,\mathrm{ }w\in {R}^{T}\right\}`
is the space of regression trees, :math:`T` is the number of
leaves in the tree, :math:`w` is a leaf weights vector, :math:`w_i` is a score
on :math:`i`-th leaf. :math:`q(x)` represents the structure of each tree that maps an
observation to the corresponding leaf index.

Training procedure is an iterative functional gradient descent
algorithm which minimizes objective function over function space by
iteratively choosing a function (regression tree) that points in the
negative gradient direction. The objective function is

.. math::
	L(f)=\sum _{i=1}^{n}l({y}_{i}, f({x}_{i})) + \sum _{k=1}^{M}\Omega({f}_{k})

where :math:`l(f)` is twice differentiable convex loss function and
:math:`\Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||` is a regularization term that penalizes the complexity of
the model defined by the number of leaves T and the L2 norm of the weights :math:`||w||` for each tree, :math:`\gamma` and :math:`\lambda`
are regularization parameters.

Training Stage
--------------

Library uses the second-order approximation of objective function

.. math::
	{L}^{(k)}(f)\approx \sum _{i=1}^{n}({g}_{i}{f}_{k}({x}_{i})+ \frac{1}{2}{h}_{i}{{f}_{k}}^{2}({x}_{i})) + \Omega (f_k),

where :math:`g_i= \frac{\partial l({y}_{i},{\hat{y_i}}^{(k-1)})}{\partial {\hat{y_i}}^{(k-1)}}`,
:math:`h_i= \frac{{\partial }^{2}l({y}_{i}, {\hat{y_i}}^{(k-1)})}{{\partial }^{2}{\hat{y_i}}^{(k-1)}}`
and following algorithmic framework for the training stage.

Let :math:`S = (X, Y)` be the set of observations. Given the training
parameters, such as the number of iterations :math:`M`, loss function :math:`l(f)`, regression tree training parameters,
regularization parameters :math:`\gamma` and :math:`\lambda`, shrinkage (learning rate) parameter :math:`\theta`, the
algorithm does the following:

- Find an initial guess :math:`\hat{y_i}^{(0)}`, :math:`i = 1, \ldots, n`

- For :math:`k = 1, \ldots , M`:

  - Update :math:`g_i` and :math:`h_i`, :math:`i = 1, \ldots, n`

  - Grow a regression tree :math:`{f}_{k}\in F` that minimizes the objective function
    :math:`-\frac{1}{2}\sum _{j=1}^{T}\frac{{G}_{j}^{2}}{{H}_{j}+\lambda }+\gamma T`, where
    :math:`G_j=\sum _{i\in {I}_{j}}{g}_{j}`, :math:`{H}_{j}=\sum _{i\in {I}_{j}}{h}_{j}`, :math:`{I}_{j}= \{i| ({x}_{i})=j\}`, :math:`j=1, \ldots, T`.

  - Assign an optimal weight :math:`{w_j}^{*}= \frac{G_j}{H_j +\lambda }` to the leaf :math:`j`, :math:`j = 1, \ldots, T`.

  - Apply shrinkage parameter :math:`\theta` to the tree leafs and add the tree to the model

  - Update :math:`\hat{y_i}^{(k)}`

The algorithm for growing the tree:

- Generate a bootstrap training set if required (stochastic
  gradient boosting) as follows: select randomly without
  replacement :math:`N = f * n` observations, where :math:`f` is a fraction of
  observations used for training of one tree.

- Start from the tree with depth :math:`0`.

- For each leaf node in the tree:

  - Choose a subset of feature for split finding if required (stochastic gradient boosting).

  - Find the best split that maximizes the gain:

    .. math::
       \frac{{G}_{L}^{2}}{{H}_{L}+\lambda }+ \frac{{G}_{R}^{2}}{{H}_{R}+\lambda }- \frac{{({G}_{L}+{G}_{R})}^{2}}{{H}_{L}+ {H}_{R}+\lambda }- \gamma  

	- Stop when a termination criterion is met.

For more details, see [Chen2016]_.

The library supports the following termination criteria when
growing the tree:

- **Minimal number of observations in a leaf node.** Node t is not
  processed if the subset of observations is smaller than the
  predefined value. Splits that produce nodes with the number of
  observations smaller than that value are not allowed.

- **Maximal tree depth.** Node t is not processed, if its depth in
  the tree reached the predefined value.

- **Minimal split loss.** Node t is not processed, if the best
  possible split is smaller than parameter :math:`\gamma`.


Prediction Stage
----------------

Given a gradient boosted trees model and vectors :math:`(x_1, \ldots, x_r)`, the problem is to calculate the responses for those
vectors. To solve the problem for each given query vector :math:`x_i`, the algorithm finds the leaf node in a tree in the
ensemble which gives the response by that tree. Resulting response
is based on an aggregation of responses from all trees in the
ensemble. For detailed definition, see description of a specific
algorithm.


Split Calculation Mode
----------------------

The library supports two split calculation modes:

- exact - all possible split values are examined when searching
  for the best split for a feature.

- inexact - continuous features are bucketed into discrete bins
  and the possible splits are restricted by the buckets borders
  only.

.. _gb_trees_batch:

Batch Processing
****************

Gradient boosted trees classification and regression follows the
general workflow described in :ref:`classification_usage_model` and :ref:`regression_usage_model`.

.. rubric:: Training

For description of the input and output, refer to .

At the training stage, the gradient boosted trees batch algorithm
has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for Gradient Boosted Trees (Batch Processing)
   :widths: 10 20 30
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``splitMethod``
     - ``inexact``
     - Split computation mode.

       Possible values:

       + ``inexact`` - continuous features are bucketed into discrete bins and the buckets borders are examined only
       + ``exact`` - all possible splits for a given feature are examined

   * - ``maxIterations``
     - :math:`50`
     - Maximal number of iterations when training the model, defines maximal number of trees in the model.
   * - ``maxTreeDepth``
     - :math:`6`
     - Maximal tree depth. If the parameter is set to :math:`0` then the depth is unlimited.
   * - ``shrinkage``
     - :math:`0.3`
     - Learning rate of the boosting procedure. Scales the contribution of each tree by a factor :math:`(0, 1]`
   * - ``minSplitLoss``
     - :math:`0`
     - Loss regularization parameter. Minimal loss reduction required to make a further partition on a leaf node of the tree. Range: :math:`[0, \infty)`
   * - ``lambda``
     - :math:`1`
     - L2 regularization parameter on weights. Range: :math:`[0, \infty)`
   * - ``observationsPerTreeFraction``
     - :math:`1`
     - Fraction of the training set S used for a single tree training, :math:`0 < \mathrm{observationsPerTreeFraction} \leq 1`. The observations are sampled randomly without replacement.
   * - featuresPerNode
     - :math:`0`
     - The number of features tried as the possible splits per node. If the parameter is set to :math:`0`, all features are used.
   * - ``minObservationsInLeafNode``
     - :math:`5`
     - Minimal number of observations in the leaf node.
   * - ``memorySavingMode``
     - ``false``
     - If true then use memory saving (but slower) mode.
   * - ``engine``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator.
   * - ``maxBins``
     - :math:`256`
     - Used with inexact split method only. Maximal number of discrete bins to
       bucket continuous features. Increasing the number results in higher
       computation costs
   * - ``minBinSize``
     - :math:`5`
     - Used with inexact split method only. Minimal number of observations in a bin.

