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

.. default-domain:: cpp

.. _alg_df:

==================================================
Decision Forest Classification and Regression (DF)
==================================================

.. include:: ../../../includes/ensembles/df-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _df_t_math:

Training
--------

Given :math:`n` feature vectors :math:`X=\{x_1=(x_{11},\ldots,x_{1p}),\ldots,x_n=(x_{n1},\ldots,x_{np})\}` of
size :math:`p`, their non-negative observation weights :math:`W=\{w_1,\ldots,w_n\}` and :math:`n` responses :math:`Y=\{y_1,\ldots,y_n\}`,

.. tabs::

  .. group-tab:: Classification

    - :math:`y_i \in \{0, \ldots, C-1\}`, where :math:`C` is the number of classes

  .. group-tab:: Regression

    - :math:`y_i \in \mathbb{R}`

the problem is to build a decision forest classification or regression model.

The library uses the following algorithmic framework for the training
stage. Let :math:`S = (X, Y)` be the set of observations. Given positive
integer parameters, such as the number of trees :math:`B`, the bootstrap
parameter :math:`N = f*n`, where :math:`f` is a fraction of observations used for
a training of each tree in the forest, and the number of features per node :math:`m`, the
algorithm does the following for :math:`b = 1, \ldots, B`:

- Selects randomly with replacement the set :math:`D_b` of :math:`N`
  vectors from the set :math:`S`. The set :math:`D_b` is called a
  *bootstrap* set.
- Trains a :ref:`decision tree <dt>` classifier :math:`T_b` on :math:`D_b`
  using parameter :math:`m` for each tree.

:ref:`Decision tree <dt>` :math:`T` is trained using the training set :math:`D` of size :math:`N`.
Each node :math:`t` in the tree corresponds to the subset :math:`D_t` of
the training set :math:`D`, with the root node being :math:`D` itself. Each
internal node :math:`t` represents a binary test (split) that divides the
subset :math:`X_t` in two subsets, :math:`X_{t_L}` and :math:`X_{t_R}`,
corresponding to their children, :math:`t_L` and :math:`t_R`.

.. _df_t_math_dense:

Training method: *Dense*
++++++++++++++++++++++++

In the *dense* training method, all possible data points for each feature are considered as possible splits for the current node
and evaluated best-split computation.

.. _df_t_math_hist:

Training method: *Hist*
+++++++++++++++++++++++

In the *hist* training method, only bins are considered for best split computation.
Bins are continuous intervals of data points for a selected feature.
They are computed for each feature during the initialization stage of the algorithm.
Each value from the initially provided data is substituted
with the value of the corresponding bin. It decreases the computational time complexity
from :math:`O(n_r \cdot n_f)` to :math:`O(n_b \cdot n_f)`,
but decreases algorithm accuracy, where :math:`n_r` is number of rows, :math:`n_b` is number of bins,
and :math:`n_f` is number of selected features.

Split strategy
++++++++++++++

.. note::
  The `random` split strategy is supported only for the `hist` method. The dense method supports only the `best` strategy.

There are two split strategies for building trees:

.. tabs::

  .. group-tab:: Best splitter

    The threshold for a node is chosen as the best among all bins and all selected features according to split criteria(see  **Split Criteria** below).
    The computational time complexity for the `best` splitter is :math:`O(n_f \cdot n_b)` for each node. The best splitting strategy
    builds a tree with optimal splits on each level.

  .. group-tab:: Random splitter

    The threshold for a node is chosen randomly for each selected feature. The split threshold is chosen as the best among all pairs
    (feature, random threshold) according to split criteria(see  **Split Criteria** below).
    The computational time complexity for the `random` splitter as :math:`O(n_f)` for each node.
    The random splitting strategy does not build a tree with optimal trees, but in the case of big tree ensembles, it provides a more robust
    model comparing to the `best` strategy.

Split Criteria
++++++++++++++

The metric for measuring the best split is called *impurity*,
:math:`i(t)`. It generally reflects the homogeneity of responses within
the subset :math:`D_t` in the node :math:`t`.

.. tabs::

  .. group-tab:: Classification

   *Gini index* is an impurity metric for classification, calculated as follows:

   .. math::
   	{I}_{Gini}\left(D\right)=1-\sum _{i=0}^{C-1}{p}_{i}^{2}

   where

   - :math:`D` is a set of observations that reach the node;
   - :math:`p_i` is specified in the table below:

   .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

   .. list-table:: Decision Forest Split Criteria Calculation
      :widths: 10 10
      :header-rows: 1
      :align: left
      :class: longtable

      * - Without sample weights
        - With sample weights
      * - :math:`p_i` is the observed fraction of observations that belong to class :math:`i` in :math:`D`
        - :math:`p_i` is the observed weighted fraction of observations that belong to class :math:`i` in :math:`D`:

          .. math::

             p_i = \frac{\sum_{d \in \{d \in D | y_d = i \}} W_d}{\sum_{d \in D} W_d}

  .. group-tab:: Regression

   *MSE* is an impurity metric for regression, calculated as follows:

   .. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

   .. list-table:: MSE Impurity Metric
      :widths: 10 10
      :header-rows: 1
      :align: left
      :class: longtable

      * - Without sample weights
        - With sample weights
      * - :math:`I_{\mathrm{MSE}}\left(D\right) = \frac{1}{W(D)} \sum _{i=1}^{W(D)}{\left(y_i - \frac{1}{W(D)} \sum _{j=1}^{W(D)} y_j \right)}^{2}`
        - :math:`I_{\mathrm{MSE}}\left(D\right) = \frac{1}{W(D)} \sum _{i \in D}{w_i \left(y_i - \frac{1}{W(D)} \sum _{j \in D} w_j y_j \right)}^{2}`
      * - :math:`W(S) = \sum_{s \in S} 1`, which is equivalent to the number of elements in :math:`S`
        - :math:`W(S) = \sum_{s \in S} w_s`

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
  Grow trees with positive maximal number of leaf nodes in a :ref:`best-first <df_t_best_first_strategy>` fashion.
  Best nodes are defined by relative reduction in impurity.
  If maximal number of leaf nodes equals zero, then this criterion does not limit the number of leaf nodes,
  and trees grow in a :ref:`depth-first <df_t_depth_first_strategy>` fashion.

Tree Building Strategies
++++++++++++++++++++++++

Maximal number of leaf nodes defines the strategy of tree building:
:ref:`depth-first <df_t_depth_first_strategy>` or :ref:`best-first <df_t_best_first_strategy>`.

.. _df_t_depth_first_strategy:

Depth-first Strategy
~~~~~~~~~~~~~~~~~~~~

If maximal number of leaf nodes equals zero, a :ref:`decision tree <dt>` is built using depth-first strategy.
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

.. _df_t_best_first_strategy:

Best-first Strategy
~~~~~~~~~~~~~~~~~~~

If maximal number of leaf nodes is positive, a :ref:`decision tree <dt>` is built using best-first strategy.
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

.. _df_i_math:

Inference
---------

Given decision forest classification or regression model and vectors :math:`x_1, \ldots, x_r`,
the problem is to calculate the responses for those vectors.

.. _df_i_math_dense_hist:

Inference methods: *Dense* and *Hist*
-------------------------------------

*Dense* and *hist* inference methods perform prediction in the same way.
To solve the problem for each given query vector :math:`x_i`, the
algorithm does the following:

.. tabs::

  .. group-tab:: Classification

    For each tree in the forest, it finds the leaf node that gives :math:`x_i` its label. The label :math:`y` that the majority of
    trees in the forest vote for is chosen as the predicted label for the query vector :math:`x_i`.

  .. group-tab:: Regression

    For each tree in the forest, it finds the leaf node that gives :math:`x_i` the response as the mean of
    dependent variables. The mean of responses from all trees in the forest is the predicted response for the query vector :math:`x_i`.

Additional Characteristics Calculated by the Decision Forest
------------------------------------------------------------

Decision forests can produce additional characteristics, such as
an estimate of generalization error and an importance measure
(relative decisive power) of each of p features (variables).

Out-of-bag Error
----------------

The estimate of the generalization error based on the training
data can be obtained and calculated as follows:

.. tabs::

  .. group-tab:: Classification

    - For each vector :math:`x_i` in the dataset :math:`X`, predict its label :math:`\hat{y_i}` by having the majority of votes from the trees that
      contain :math:`x_i` in their OOB set, and vote for that label.

    - Calculate the OOB error of the decision forest :math:`T` as the average
      of misclassifications:

      .. math::
         OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{y_i \in {D}^{\text{'}}}I\{y_i \ne \hat{y_i}\}\text{,where }{D}^{\text{'}}={\bigcup }_{b=1}^{B}\overline{D_b}.

    - If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`: :math:`OOB(x_i) = I\{{y}_{i}\ne \hat{{y}_{i}}\}`

  .. group-tab:: Regression

    - For each vector :math:`x_i` in the dataset :math:`X`, predict its response :math:`\hat{y_i}`
      as the mean of prediction from the trees that contain :math:`x_i` in their OOB set:

      :math:`\hat{y_i} = \frac{1}{{|B}_{i}|}\sum _{b=1}^{|B_i|}\hat{y_{ib}}`, where :math:`B_i= \bigcup{T_b}: x_i \in \overline{D_b}` and :math:`\hat{y_{ib}}` is the result of prediction
      :math:`x_i` by :math:`T_b`.

    - Calculate the OOB error of the decision forest :math:`T` as the Mean-Square Error (MSE):

      .. math::
         OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{{y}_{i} \in {D}^{\text{'}}}\sum {(y_i-\hat{y_i})}^{2}, \text{where } {D}^{\text{'}}={\bigcup}_{b=1}^{B}\overline{{D}_{b}}

    - If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`:

      .. math::
         OOB(x_i) = {(y_i-\hat{y_i})}^{2}

Variable Importance
-------------------

There are two main types of variable importance measures:

- *Mean Decrease Impurity* importance (MDI)

  Importance of the :math:`j`-th variable for predicting :math:`Y` is the sum of
  weighted impurity decreases :math:`p(t) \Delta i(s_t, t)` for all nodes
  :math:`t` that use :math:`x_j`, averaged over all :math:`B` trees in the
  forest:

  .. math::
    MDI\left(j\right)=\frac{1}{B}\sum _{b=1}^{B} \sum _{t\in {T}_{b}:v\left({s}_{t}\right)=j}p\left(t\right)\Delta i\left({s}_{t},t\right),

  where :math:`p\left(t\right)=\frac{|{D}_{t}|}{|D|}` is the fraction of observations reaching node :math:`t`
  in the tree :math:`T_b`, and :math:`v(s_t)` is the index of the
  variable used in split :math:`s_t`.

- *Mean Decrease Accuracy* (MDA)

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

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Decision Forest Classification and Regression <api_df>`.

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SMPD mode (only on GPU).

