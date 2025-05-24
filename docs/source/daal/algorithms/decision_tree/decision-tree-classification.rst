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

.. _dt_classification:

Classification Decision Tree
****************************

.. toctree::
   :maxdepth: 1


Classification decision tree is a kind of a decision tree described in :ref:`decision_tree`.

Details
=======

Given:

- n feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of size :math:`p`
- The vector of class labels :math:`y = (y_1, \ldots, y_n)` that describes the
  class to which the feature vector :math:`x_i` belongs, where :math:`y_i \in \{0, 1, \ldots, C-1\}` and C is the number of classes.

The problem is to build a decision tree classifier.

Split Criteria
--------------

The library provides the decision tree classification algorithm
based on split criteria Gini index [Breiman84]_ and Information gain [Quinlan86]_, [Mitchell97]_:

#.

 Gini index

 .. math::
	 {I}_{Gini}\left(D\right)=1-\sum _{i=0}^{C-1}{p}_{i}^{2}


 where

 - :math:`D` is a set of observations that reach the node

 - :math:`p_i` is the observed fraction of observations with class :math:`i` in :math:`D`

 To find the best test using Gini index, each possible test is examined using

 .. math::
	 \text{Δ}{I}_{Gini}\left(D,\text{ τ}\right)={I}_{Gini}\left(D\right)-\sum _{v\in O\left(\text{τ}\right)}\frac{|{D}_{v}|}{|D|}{I}_{Gini}\left({D}_{v}\right)\phantom{\rule{0ex}{0ex}}\phantom{\rule{0ex}{0ex}}


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

Training Stage
--------------

The classification decision tree follows the algorithmic framework
of decision tree training described in :ref:`decision_tree`.

Prediction Stage
----------------

The classification decision tree follows the algorithmic framework
of decision tree prediction described in :ref:`decision_tree`.

Given decision tree and vectors :math:`x_i, \ldots, x_r`, the
problem is to calculate the responses for those vectors.

Batch Processing
================

Decision tree classification follows the general workflow described
in :ref:`classification_usage_model`.

Training
--------

In addition to common input for a classifier, decision trees can accept the following
inputs that are used for post-pruning:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Training Input for Decision Tree Classification (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60
   :class: longtable

   * - Input ID
     - Input
   * - ``dataForPruning``
     - Pointer to the :math:`m \times p` numeric table with the pruning data set. This table
       can be an object of any class derived from NumericTable.
   * - ``labelsForPruning``
     - Pointer to the :math:`m \times 1` numeric table with class labels. This table can be
       an object of any class derived from NumericTable except
       PackedSymmetricMatrix and PackedTriangularMatrix.


At the training stage, decision tree classifier has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Training Parameters for Decision Tree Classification (Batch Processing)
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
     - The computation method used by the decision tree classification. The
       only training method supported so far is the default dense method.
   * - ``nClasses``
     - Not applicable
     - The number of classes. A required parameter.
   * - ``splitCriterion``
     - ``infoGain``
     - Split criterion to choose the best test for split nodes. Available split criteria for decision trees:

       + ``gini`` - the Gini index
       + ``infoGain`` - the information gain

   * - ``pruning``
     - ``reducedErrorPruning``
     - Method to perform post-pruning. Available options for the pruning parameter:

       + ``reducedErrorPruning`` - reduced error pruning. Provide dataForPruning and labelsForPruning inputs, if you use pruning.
       + ``none`` - do not prune.

   * - ``maxTreeDepth``
     - :math:`0`
     - Maximum tree depth. Zero value means unlimited depth. Can be any non-negative number.
   * - ``minObservationsInLeafNodes``
     - :math:`1`
     - Minimum number of observations in the leaf node. Can be any positive number.


Prediction
----------

At the prediction stage, decision tree classifier has the following
parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Prediction Parameters for Decision Tree Classification (Batch Processing)
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
     - The computation method used by the decision tree classification. The
       only training method supported so far is the default dense method.


Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`dt_cls_dense_batch.cpp <decision_tree/dt_cls_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`decision_tree_classification.py`
    - :daal4py_example:`decision_tree_classification_traverse.py`
