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

.. _decision_forest:

Decision Forest
================

Details
*******

Given n feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}` of
:math:`n` :math:`p`-dimensional feature vectors and n responses :math:`y = (y_1, \ldots, y_n)`, the problem is to build a decision forest
classification or regression model.

Training Stage
--------------

Library uses the following algorithmic framework for the training
stage. Let :math:`S = (X, Y)` be the set of observations. Given a positive
integer parameters, such as the number of trees :math:`B`, the bootstrap
parameter :math:`N = f*n`, where :math:`f` is a fraction of observations used for
a training of one tree, and the number of features per node :math:`m`, the
algorithm does the following for :math:`b = 1, \ldots ,B`:

- Selects randomly with replacement the set :math:`D_b` of :math:`N`
  vectors from the set :math:`S`. The set :math:`D_b` is called a
  *bootstrap* set.
- Trains a decision tree classifier :math:`T_b` on :math:`D_b`
  using parameter :math:`m` for each tree.

Decision tree :math:`T` is trained using the training set :math:`D` of size :math:`N`.
Each node :math:`t` in the tree corresponds to the subset :math:`D_t` of
the training set :math:`D`, with the root node being :math:`D` itself. Its
internal nodes :math:`t` represent a binary test (split) dividing their
subset :math:`X_t` in two subsets :math:`X_{t_L}` and :math:`X_{t_R}`,
corresponding to their children :math:`t_L` and :math:`t_R`.

.. _df_inexact_hist_method:

Inexact Histogram Computation Method
++++++++++++++++++++++++++++++++++++

In inexact histogram method only a selected subset of splits is considered for computation of a best split.
This subset is computed for each feature at the initialization stage of the algorithm. After the set of splits is computed,
each value from initially provided data is substituted with the value of the corresponding bin.
The bins are continuous intervals between the selected splits.

Split Criteria
++++++++++++++

The metric for measuring the best split is called *impurity*,
:math:`i(t)`. It generally reflects the homogeneity of responses within
the subset :math:`D_t` in the node :math:`t`.
For the detailed definition of :math:`i(t)` metrics, see the description
of a specific algorithm.

Let the *impurity decrease* in the node :math:`t` be

.. math::
	\Delta i\left(t\right)=i\left(t\right)–\frac{|{D}_{t}{}_{{}_{L}}|}{|{D}_{t}|}i\left({t}_{L}\right)–\frac{|{D}_{t}{}_{{}_{R}}|}{|{D}_{t}|}i\left({t}_{R}\right).\text{ }

Termination Criteria
++++++++++++++++++++

The library supports the following termination criteria of
decision forest training:

Minimal number of observations in a leaf node
  Node :math:`t` is not processed if :math:`|D_t|` is smaller than the predefined value.
  Splits that produce nodes with the number of observations smaller than that value are not allowed.

Minimal number of observations in a split node
  Node :math:`t` is not processed if :math:`|D_t|` is smaller than the predefined value.
  Splits that produce nodes with the number of observations smaller than that value are not allowed.

Minimum weighted fraction of the sum total of weights of all the input observations required to be at a leaf node
  Node :math:`t` is not processed if :math:`|D_t|` is smaller than the predefined value.
  Splits that produce nodes with the number of observations smaller than that value are not allowed.

Maximal tree depth
  Node :math:`t` is not processed if its depth in the tree reached the predefined value.

Impurity threshold
  Node :math:`t` is not processed if its impurity is smaller than the predefined threshold.

Maximal number of leaf nodes
  Grow trees with positive maximal number of leaf nodes in a :ref:`best-first <best_first_strategy>` fashion.
  Best nodes are defined by relative reduction in impurity.
  If maximal number of leaf nodes equals zero, then this criterion does not limit the number of leaf nodes,
  and trees grow in a :ref:`depth-first <depth_first_strategy>` fashion.

Tree Building Strategies
++++++++++++++++++++++++

Maximal number of leaf nodes defines the strategy of tree building:
:ref:`depth-first <depth_first_strategy>` or :ref:`best-first <best_first_strategy>`.

.. _depth_first_strategy:

Depth-first Strategy
~~~~~~~~~~~~~~~~~~~~

If maximal number of leaf nodes equals zero, a decision tree is built using depth-first strategy.
In each terminal node :math:`t`, the following recursive procedure is applied:

- Stop if the termination criteria are met.
- Choose randomly without replacement :math:`m` feature indices :math:`J_t \in \{0, 1, \ldots, p-1\}`.
- For each :math:`j \in J_t`, find the best split :math:`s_{j,t}` that
  partitions subset :math:`D_t` and maximizes impurity decrease
  :math:`\Delta i(t)`.
- A node is a split if this split induces a decrease of the impurity greater than or equal to the predefined value.
  Get the best split :math:`s_t` that maximizes impurity decrease
  :math:`\Delta i` in all :math:`s_{j,t}` splits.
- Apply this procedure recursively to :math:`t_L` and :math:`t_R`.

.. _best_first_strategy:

Best-first Strategy
~~~~~~~~~~~~~~~~~~~

If maximal number of leaf nodes is positive, a decision tree is built using best-first strategy.
In each terminal node :math:`t`, the following steps are applied:

- Stop if the termination criteria are met.
- Choose randomly without replacement :math:`m` feature indices :math:`J_t \in \{0, 1, \ldots, p-1\}`.
- For each :math:`j \in J_t`, find the best split :math:`s_{j,t}` that
  partitions subset :math:`D_t` and maximizes impurity decrease
  :math:`\Delta i(t)`.
- A node is a split if this split induces a decrease of the impurity greater than or equal to the predefined value
  and the number of split nodes is less or equal to :math:`\mathrm{maxLeafNodes} – 1`.
  Get the best split :math:`s_t` that maximizes impurity decrease
  :math:`\Delta i` in all :math:`s_{j,t}` splits.
- Put a node into a sorted array, where sort criterion is the improvement in impurity :math:`\Delta i(t)|D_t|`.
  The node with maximal improvement is the first in the array. For a leaf node, the improvement in impurity is zero.
- Apply this procedure to :math:`t_L` and :math:`t_R` and grow a tree one by one getting the first element from the array
  until the array is empty.


Random Numbers Generation
+++++++++++++++++++++++++

To create a *bootstrap* set and choose feature indices in the
performant way, the training algorithm requires the source of
random numbers, capable to produce sequences of random numbers in
parallel.

Initialization of the engine in the decision forest is based on
the scheme below:

The state of the engine is updated once the training of the
decision forest model is completed. The library provides support
to retrieve the instance of the engine with updated state that can
be used in other computations. The update of the state is
engine-specific and depends on the parallelization technique used
as defined earlier:

- Family: the updated state is the set of states that represent
  individual engines in the family.

- Leapfrog: the updated state is the state of the sequence with
  the rightmost position on the sequence. The example below
  demonstrates the idea for case of 2 subsequences (‘x’ and ‘o’)
  of the random number sequence:

-  SkipAhead: the updated state is the state of the independent
   sequence with the rightmost position on the sequence. The
   example below demonstrates the idea for case of 2 subsequences
   (‘x’ and ‘o’) of the random number sequence:

Prediction Stage
-----------------

Given decision forest classifier and vectors :math:`x_1, \ldots, x_r`,
the problem is to calculate the responses for those vectors. To
solve the problem for each given query vector :math:`x_i` , the
algorithm finds the leaf node in a tree in the forest that gives
the response by that tree. The response of the forest is based on
an aggregation of responses from all trees in the forest. For the
detailed definition, see the description of a specific algorithm.

Additional Characteristics Calculated by the Decision Forest
------------------------------------------------------------

Decision forests can produce additional characteristics, such as
an estimate of generalization error and an importance measure
(relative decisive power) of each of p features (variables).

Out-of-bag Error
----------------

The estimate of the generalization error based on the training
data can be obtained and calculated as follows:

-  For each tree :math:`T_b` in the forest, trained on the bootstrap set :math:`D_b`, the set :math:`\overline{D_b}=S \setminus {D}_{b}` is called the out-of-bag (OOB) set.
-  Predict the data from :math:`\overline{{D}_{b}}` set by :math:`T_b`.
-  For each vector :math:`x_i` in the dataset X, predict its response :math:`\hat{y_i}` by the trees that contain :math:`x_i` in their OOB set.
-  Aggregate the out-of-bag predictions in all trees and calculate the OOB error of the decision forest.
-  If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`.

For the detailed definition, see the description of a specific
algorithm.

Variable Importance
-------------------

There are two main types of variable importance measures:

- *Mean Decrease Impurity* importance (MDI).

  Importance of the :math:`j`-th variable for predicting :math:`Y` is the sum of
  weighted impurity decreases :math:`p(t) \Delta i(s_t, t)` for all nodes
  :math:`t` that use :math:`x_j`, averaged over all :math:`B` trees in the
  forest:

  .. math::
    MDI\left(j\right)=\frac{1}{B}\sum _{b=1}^{B} \sum _{t\in {T}_{b}:v\left({s}_{t}\right)=j}p\left(t\right)\Delta i\left({s}_{t},t\right),

  where :math:`p\left(t\right)=\frac{|{D}_{t}|}{|D|}` is the fraction of observations reaching node :math:`t`
  in the tree :math:`T_b`, and :math:`v(s_t)` is the index of the
  variable used in split :math:`s_t` .

- *Mean Decrease Accuracy* (MDA).

  Importance of the :math:`j`-th variable for predicting :math:`Y` is the average
  increase in the OOB error over all trees in the forest when the
  values of the :math:`j`-th variable are randomly permuted in the OOB
  set. For that reason, this latter measure is also known as
  *permutation importance*.

  In more details, the library calculates MDA importance as
  follows:

  - Let :math:`\pi (X,j)` be the set of feature vectors where the :math:`j`-th variable is randomly permuted over all vectors in the set.
  - Let :math:`E_b` be the OOB error calculated for :math:`T_b:` on its out-of-bag dataset :math:`\overline{D_b}`.
  - Let :math:`E_{b,j}` be the OOB error calculated for :math:`T_b:` using :math:`\pi \left(\overline{{X}_{b}},j\right)`, and its out-of-bag dataset :math:`\overline{D_b}` is permuted on the :math:`j`-th variable. Then

    * :math:`{\delta }_{b,j}={E}_{b}-{E}_{b,j}` is the OOB error increase for the tree :math:`T_b`.
    * :math:`Raw MDA\left(j\right)=\frac{1}{B}\sum _{b=1}^{B}{\delta }_{b,j}` is MDA importance.
    * :math:`Scaled MDA\left(j\right)=\frac{Raw MDA\left({x}_{j}\right)}{\frac{{\sigma }_{j}}{\sqrt{B}}}`, where :math:`{\sigma }_{j}^{2}` is the variance of :math:`D_{b,j}`

.. _df_batch:

Batch Processing
****************

Decision forest classification and regression follows the general
workflow described in :ref:`classification_usage_model`.

Training
--------

At the training stage, decision forest regression has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Training Parameters for Decision Forest (Batch Processing)
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
   * - ``binningStrategy``
     - ``quantiles``
     - Used only with ``method="hist"`` and CPU only. Selects the algorithm
       used to calculate bin edges. ``quantiles`` results in bins with a similar
       amount of training data points. ``averages`` divides the range of values
       observed in the training data set into equal-width bins of size `(max -
       min) / maxBins`.
   * - ``bootstrap``
     - ``true``
     - If true, the training set for a tree is a bootstrap of the whole training set.
       If false, the whole training set is used to build trees.
   * - ``engine``
     - ``SharePtr<engines::mt2203::Batch>()>``
     - Pointer to the random number generator engine.

       The random numbers produced by this engine are used to choose the bootstrap set,
       split features in every split node in a tree, and generate permutation required in computations
       of MDA variable importance.
   * - ``featuresPerNode``
     - :math:`0`
     - The number of features tried as possible splits per node. If the
       parameter is set to :math:`0`, the library uses the square root of the number of
       features, :math:`\sqrt{p}`, for classification
       and :math:`\frac{p}{3}` features for regression.
   * - ``impurityThreshold``
     - :math:`0`
     - The threshold value used as stopping criteria: if the impurity value in
       the node is smaller than the threshold, the node is not split anymore.
   * - ``maxBins``
     - :math:`256`
     - Used only with ``method="hist"``. Maximal number of discrete bins to
       bucket continuous features. Increasing the number results in higher
       computation costs. Selecting :math:`0` disables creating buckets.
   * - ``maxLeafNodes``
     - :math:`0`
     - Grow trees with positive maximal number of leaf nodes in a :ref:`best-first <best_first_strategy>` fashion.
       Best nodes are defined as relative reduction in impurity.
       If maximal number of leaf nodes equals zero,
       then this parameter does not limit the number of leaf nodes, and trees grow in a :ref:`depth-first <depth_first_strategy>` fashion.
   * - ``maxTreeDepth``
     - :math:`0`
     - Maximal tree depth. Default is :math:`0` (unlimited).
   * - ``method``
     - ``defaultDense``
     - The computation method used by the decision forest classification.

       For CPU:

       - ``defaultDense`` - default performance-oriented method
       - ``hist`` - inexact histogram computation method

       For GPU:

       - ``hist`` - :ref:`inexact histogram computation method <df_inexact_hist_method>`
   * - ``minBinSize``
     - :math:`5`
     - Used only with ``method="hist"``. Minimal number of observations in a bin.
   * - ``minImpurityDecreaseInSplitNode``
     - :math:`0.0`
     - Minimum amount of impurity decrease required to split a node; it can be any non-negative number.
   * - ``minObservationsInLeafNode``
     - :math:`1` for classification, :math:`5` for regression
     - Minimum number of observations in the leaf node.
   * - ``minObservationsInSplitNode``
     - :math:`2`
     - Minimum number of samples required to split an internal node; it can be any non-negative number.
   * - ``minWeightFractionInLeafNode``
     - :math:`0.0`
     - Minimum weighted fraction of the sum total of weights of all the input observations required to be at a leaf node,
       from :math:`0.0` to :math:`0.5`.

       All observations have equal weights if the weights of the observations are not provided.
   * - ``nTrees``
     - :math:`100`
     - The number of trees in the forest.
   * - ``observationsPerTreeFraction``
     - :math:`1`
     - Fraction of the training set S used to form the bootstrap set for a
       single tree training, :math:`0 < \mathrm{observationsPerTreeFraction} \leq 1`. The
       observations are sampled randomly with replacement.
   * - ``resultsToCompute``
     - :math:`0`
     - The 64-bit integer flag that specifies which extra characteristics of
       the decision forest to compute. Provide one of the following values to
       request a single characteristic or use bitwise OR to request a
       combination of the characteristics:

       + ``computeOutOfBagError``
       + ``computeOutOfBagErrorPerObservation``
   * - **DEPRECATED:** ``seed``
     - :math:`777`
     - The seed for random number generator, which is used to choose the
       bootstrap set, split features in every split node in a tree, and
       generate permutation required in computations of ``MDA`` variable importance.

       .. note::

          This parameter is deprecated and will be removed in future releases. Use ``engine`` instead.
   * - ``varImportance``
     - ``none``
     - The variable importance computation mode.

       Possible values:

       + ``none`` – variable importance is not calculated
       + ``MDI`` - Mean Decrease of Impurity, also known as the Gini importance or Mean Decrease Gini
       + ``MDA_Raw`` - Mean Decrease of Accuracy (permutation importance)
       + ``MDA_Scaled`` - the MDA_Raw value scaled by its standard deviation



Output
------

In addition to regression or classifier output, decision forest
calculates the result described below. Pass the ``Result ID`` as a
parameter to the methods that access the result of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Output for Decision Forest (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``outOfBagError``
     - A numeric table :math:`1 \times 1` containing out-of-bag error computed when the
       ``computeOutOfBagErroroption`` option is on.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``.
   * - ``variableImportance``
     - A numeric table :math:`1 \times p` that contains variable importance values for each
       feature. If you set the ``varImportance`` parameter to none, the library
       returns a null pointer to the table.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix`` and ``PackedSymmetricMatrix``.
   * - ``outOfBagErrorPerObservation``
     - A numeric table of size :math:`1 \times n` that contains the computed out-of-bag error
       when the ``computeOutOfBagErrorPerObservation`` option is enabled. The value
       :math:`-1` in the table indicates that no OOB value was computed because this
       observation was not in OOB set for any of the trees in the model (never
       left out during the bootstrap).

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``.
   * - ``updatedEngine``
     - Engine instance with state updated after computations.


Performance Considerations
**************************

To get the best performance of the decision forest variable
importance computation, use the Mean Decrease Impurity (MDI) rather
than the Mean Decrease Accuracy (MDA) method.

.. include:: ../../../opt-notice.rst