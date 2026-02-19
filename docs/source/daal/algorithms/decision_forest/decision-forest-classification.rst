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

.. _df_classification:

Classification Decision Forest
------------------------------

.. toctree::
   :glob:
   :maxdepth: 4

Decision forest classifier is a special case of the :ref:`decision_forest` model.

Details
=======

Given:

- :math:`n` feature vectors :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np}) \}`
  of size :math:`p`;
- their non-negative sample weights :math:`w = (w_1, \ldots, w_n)`;
- the vector of class labels :math:`y = (y_1, \ldots, y_n)` that describes the class
  to which the feature vector :math:`x_i` belongs,
  where :math:`y_i \in \{0, 1, \ldots, C-1\}` and :math:`C` is the number of classes.

The problem is to build a decision forest classifier.

Training Stage
**************

Decision forest classifier follows the algorithmic framework of
decision forest training with Gini impurity metrics as impurity
metrics [Breiman84]_.
If sample weights are provided as input, the library uses a weighted version of the algorithm.

Gini index is an impurity metric, calculated as follows:

.. math::
	{I}_{Gini}\left(D\right)=1-\sum _{i=0}^{C-1}{p}_{i}^{2}

where

- :math:`D` is a set of observations that reach the node;
- :math:`p_i` is specified in the table below:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Decision Forest Classification: impurity calculations
   :widths: 10 10
   :header-rows: 1
   :align: left

   * - Without sample weights
     - With sample weights
   * - :math:`p_i` is the observed fraction of observations that belong to class :math:`i` in :math:`D`
     - :math:`p_i` is the observed weighted fraction of observations that belong to class :math:`i` in :math:`D`:

       .. math::

          p_i = \frac{\sum_{d \in \{d \in D | y_d = i \}} W_d}{\sum_{d \in D} W_d}

Prediction Stage
****************

Given decision forest classifier and vectors :math:`x_1, \ldots, x_r`,
the problem is to calculate the labels for those
vectors. To solve the problem for each given query vector :math:`x_i`,
the algorithm finds the leaf node in a tree in the
forest that gives the classification response by that tree. The
forest chooses the label y taking the majority of trees in the
forest voting for that label.

Out-of-bag Error
****************

Decision forest classifier follows the algorithmic framework for
calculating the decision forest out-of-bag (OOB) error, where
aggregation of the out-of-bag predictions in all trees and
calculation of the OOB error of the decision forest is done as
follows:

-

 For each vector :math:`x_i` in the dataset :math:`X`, predict its label :math:`\hat{y_i}` by having the majority of votes from the trees that
 contain :math:`x_i` in their OOB set, and vote for that label.

-

 Calculate the OOB error of the decision forest :math:`T` as the average
 of misclassifications:

 .. math::
	OOB(T) = \frac{1}{|{D}^{\text{'}}|}\sum _{y_i \in {D}^{\text{'}}}I\{y_i \ne \hat{y_i}\}\text{,where }{D}^{\text{'}}={\bigcup }_{b=1}^{B}\overline{D_b}.

-  If OOB error value per each observation is required, then calculate the prediction error for :math:`x_i`: :math:`OOB(x_i) = I\{{y}_{i}\ne \hat{{y}_{i}}\}`


Variable Importance
*******************

The library computes *Mean Decrease Impurity* (MDI) importance
measure, also known as the *Gini importance* or *Mean Decrease
Gini*, by using the Gini index as impurity metrics.

Usage of Training Alternative
=============================

To build a Decision Forest Classification model using methods of the Model Builder class of Decision Forest Classification,
complete the following steps:

- Create a Decision Forest Classification model builder using a constructor with the required number of classes and trees.
- Create a decision tree and add nodes to it:

  - Use the ``createTree`` method with the required number of nodes in a tree and a label of the class for which the tree is created.
  - Use the ``addSplitNode`` and ``addLeafNode`` methods to add split and leaf nodes to the created tree.
    See the note below describing the decision tree structure.
  - After you add all nodes to the current tree, proceed to creating the next one in the same way.

- Use the ``getModel`` method to get the trained Decision Forest Classification model after all trees have been created.

.. note::

  Each tree consists of internal nodes (called non-leaf or split nodes) and external nodes (leaf nodes).
  Each split node denotes a feature test that is a Boolean expression, for example,
  f < ``featureValue`` or f = ``featureValue``, where f is a feature and ``featureValue`` is a constant.
  The test type depends on the feature type: continuous, categorical, or ordinal.

  The inducted decision tree is a binary tree, meaning that each non-leaf node has exactly two branches: true and false.
  Each split node contains ``featureIndex``, the index of the feature used for the feature test in this node, and ``featureValue``,
  the constant for the Boolean expression in the test. Each leaf node contains a ``classLabel``, the predicted class for this leaf.

  Add nodes to the created tree in accordance with the pre-calculated structure of the tree.
  Check that the leaf nodes do not have children nodes and that the splits have exactly two children.

Split Criteria
--------------

The library provides the decision tree classification algorithm
based on split criteria Gini index [Breiman84]_ and Information gain
[Quinlan86]_, [Mitchell97]_.

#.

 Gini index

 .. math::
	 {I}_{Gini}\left(D\right)=1-\sum _{i=0}^{C-1}{p}_{i}^{2}


 where

 - :math:`D` is a set of observations that reach the node

 - :math:`p_i` is the observed fraction of observations with class :math:`i` in :math:`D`

 To find the best test using Gini index, each possible test is examined using

 .. math::
	 \text{Δ}{I}_{Gini}\left(D,\text{ τ}\right)={I}_{Gini}\left(D\right)-\sum _{v\in O\left(\text{τ}\right)}\frac{|{D}_{v}|}{|D|}{I}_{Gini}\left({D}_{v}\right)\phantom{\rule{0ex}{0ex}}\phantom{\rule{0ex}{0ex}}


 where

 - :math:`O(\tau)` is the set of all possible outcomes of test :math:`\tau`

 - :math:`D_v` is the subset of :math:`D`, for which outcome of :math:`\tau` is :math:`v`, for example :math:`{D}_{v}=\left\{d\in D\text{|τ}\left(d\right)=v\right\}`

 The test to be used in the node is selected as :math:`\underset{\tau }{\text{argmax}}\text{Δ}{I}_{Gini}\left(D,\tau \right)`.
 For binary decision tree with 'true' and 'false' branches, :math:`\text{Δ}{I}_{Gini}\left(D, \text{τ}\right)={I}_{Gini}\left(D\right)-\frac{|{D}_{true}|}{|D|}{I}_{Gini}\left({D}_{true}\right)-\frac{|{D}_{false}|}{|D|}{I}_{Gini}\left({D}_{false}\right)`

#. Information gain


   .. math::
      \text{Δ}{I}_{Gini}\left(D, \text{τ}\right)={I}_{Gini}\left(D\right)-\frac{|{D}_{true}|}{|D|}{I}_{Gini}\left({D}_{true}\right)-\frac{|{D}_{false}|}{|D|}{I}_{Gini}\left({D}_{false}\right)


   where

   - :math:`O(\tau)`, :math:`D`, :math:`D_v` are defined above
   - :math:`{I}_{Entropy}\left(D\right)=-\sum _{i=0}^{C-1}{p}_{i}\mathrm{log}{p}_{i}`, with :math:`p_i` defined above in Gini index.

   Similarly to Gini index, the test to be used in the node is selected as :math:`\underset{\tau }{\text{argmax}}InfoGain\left(D,\tau \right)`. For binary decision tree with 'true' and 'false' branches, :math:`\text{Δ}{I}_{Gini}\left(D, \text{τ}\right)={I}_{Gini}\left(D\right)-\frac{|{D}_{true}|}{|D|}{I}_{Gini}\left({D}_{true}\right)-\frac{|{D}_{false}|}{|D|}{I}_{Gini}\left({D}_{false}\right)`


Types of Tests
--------------

The library inducts decision trees with the following types of
tests:

#.

   For continuous features, the test has a form of :math:`f_j < constant`, where :math:`f_j` is a feature, :math:`j \in \{1, \ldots, p\}`.

   While enumerating all possible tests for each continuous
   feature, the *constant* can be any threshold as midway between
   sequential values for sorted unique values of given feature :math:`f_j`
   that reach the node.

#.

   For categorical features, the test has a form of :math:`f_j = constant`,
   where :math:`f_j` is a feature, :math:`j \in \{1, \ldots, p\}`.

   While enumerating all possible tests for each categorical
   feature, the *constant* can be any value of given feature :math:`f_j` that reach the node.

#.

   For ordinal features, the test has a form of :math:`f_j <> constant`
   where :math:`f_j` is a feature, :math:`j \in \{1, \ldots, p\}`.

   While enumerating all possible tests for each ordinal feature,
   the *constant* can be any unique value except for the first one
   (in the ascending order) of given feature :math:`f_j` that reach
   the node

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    - :cpp_example:`df_cls_dense_batch_model_builder.cpp <decision_forest/df_cls_dense_batch_model_builder.cpp>`
    - :cpp_example:`df_cls_traversed_model_builder.cpp <decision_forest/df_cls_traversed_model_builder.cpp>`

  .. tab:: Python*

    - :daal4py_example:`decision_forest_classification_default_dense.py`
    - :daal4py_example:`decision_forest_classification_traverse.py`

Batch Processing
================

Decision forest classification follows the general workflow described in :ref:`decision_forest` and :ref:`classification_usage_model`.

Training
********

In addition to the parameters of a classifier (see :ref:`classification_usage_model`) and decision forest parameters
described in :ref:`df_batch`, the training algorithm for decision forest classification has the
following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Training Parameters for Decision Forest Classification (Batch Processing)
   :widths: 10 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.


Output
******

Decision forest classification calculates the result of regression
and decision forest. For more details, refer to :ref:`df_batch` and :ref:`classification_usage_model`.

Prediction
**********

For the description of the input and output, refer to :ref:`classification_usage_model`.

In addition to the parameters of a classifier, decision forest
classification has the following parameters at the prediction stage:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Decision Forest Classification (Batch Processing)
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
     - The computation method used by the decision forest classification. The
       only prediction method supported so far is the default dense method.
   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.
   * - ``votingMethod``
     - ``weighted``
     - A flag that specifies which method is used to compute probabilities and class labels:

       weighted
         - Probability for each class is computed as a sample mean of estimates across all trees, where each estimate
           is the normalized number of training samples for this class that were recorded in a particular leaf node for current input.
         - The algorithm returns the label for the class that gets the maximal value in a sample mean.

       unweighted
         - Probabilities are computed as normalized votes distribution across all trees of the forest.
         - The algorithm returns the label for the class that gets the majority of votes across all trees of the forest.


Examples
********

.. tabs::

  .. tab:: oneAPI DPC++

    Batch Processing:

    - :ref:`dpc_df_cls_hist_batch.cpp`

  .. tab:: oneAPI C++

    Batch Processing:

    - :ref:`cpp_df_cls_dense_batch.cpp`

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`df_cls_default_dense_batch.cpp <decision_forest/df_cls_default_dense_batch.cpp>`
    - :cpp_example:`df_cls_hist_dense_batch.cpp <decision_forest/df_cls_hist_dense_batch.cpp>`
    - :cpp_example:`df_cls_traverse_model.cpp <decision_forest/df_cls_traverse_model.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`decision_forest_classification_default_dense.py`
    - :daal4py_example:`decision_forest_classification_hist.py`
    - :daal4py_example:`decision_forest_classification_traverse.py`

