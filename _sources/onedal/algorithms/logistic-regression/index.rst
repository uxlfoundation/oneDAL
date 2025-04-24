.. Copyright 2023 Intel Corporation
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

.. highlight:: cpp
.. default-domain:: cpp

.. _alg_logreg:

===================
Logistic Regression
===================

This chapter describes the Logistic Regression algorithm implemented in |short_name|.

.. include:: ../../../includes/logistic-regression/logistic-regression-introduction.rst

------------------------
Mathematical Formulation
------------------------

.. _logreg_t_math:

Training
--------

Given :math:`n` feature vectors :math:`X=\{x_1=(x_{11},\ldots,x_{1p}),\ldots,
x_n=(x_{n1},\ldots,x_{np})\}` of size :math:`p`
and :math:`n` responses :math:`Y=\{y_1,\ldots,y_n\} \in \{0,1\}`,
the problem is to fit the model weights :math:`w=\{w_0, \ldots, w_p\}` to minimize
Logistic Loss :math:`L(X, w, y) = \sum_{i = 1}^{n} -y_i \log(prob_i) - (1 - y_i) \log(prob_i)`.
Where
* :math:`prob_i = \sigma(w_0 + \sum_{j=1}^{p} w_j x_{i, j})` - predicted probabilities,
* :math:`\sigma(x) = \frac{1}{1 + \exp(-x)}` - a sigmoid function.
Note that probabilities are binded to interval :math:`[\epsilon, 1 - \epsilon]` to avoid problems with
computing log function (:math:`\epsilon=10^{-7}` if float type is used and :math:`10^{-15}` otherwise)

.. note:: The probabilities are constrained to the interval :math:`[\epsilon, 1 - \epsilon]` to prevent issues
    when computing the logarithm function. Where :math:`\epsilon=10^{-7}` for float type and :math:`10^{-15}` otherwise.

.. _logreg_t_math_dense_batch:

Training Method: *dense_batch*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since Logistic Loss is a convex function, you can use one of the iterative solvers designed for convex problems for minimization.
During training, the data is divided into batches, and the gradients from each batch are summed up.

Refer to :ref:`Mathematical formulation: Newton-CG <newton_cg_c_math>`.

.. _logreg_t_math_sparse:

Training Method: *sparse*
~~~~~~~~~~~~~~~~~~~~~~~~~

Using this method you can train Logistic Regression model on sparse data. All you need is to provide matrix with feature vectors as
sparse table. Find more info about sparse tables here :ref:`Compressed Sparse Rows (CSR) Table: <csr_table>`.

.. _logreg_i_math:

Inference
---------

Given :math:`r` feature vectors :math:`X=\{x_1=(x_{11},\ldots,x_{1p}),\ldots,
x_r=(x_{r1},\ldots,x_{rp})\}` of size :math:`p`, the problem is to calculate the probabilities of
associated with these feature vectors belonging to each class and determine the most probable class
label for each object.

.. _logreg_i_math_dense_batch:

The probabilities are calculated using this formula
:math:`prob_i = \sigma(w_0 + \sum_{j=1}^{p} w_j x_{i, j})`.
Where :math:`\sigma(x) = \frac{1}{1 + \exp(-x)}` is a sigmoid function.
If the probability is bigger than :math:`0.5` then class label is set to :math:`1`,
otherwise to :math:`0`.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Logistic Regression <api_logreg>`.


.. rubric:: Examples: Logistic Regression

.. include:: ../../../includes/logistic-regression/logistic-regression-examples.rst
