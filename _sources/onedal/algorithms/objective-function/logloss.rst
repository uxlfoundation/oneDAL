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

.. default-domain:: cpp

.. _alg_logloss:

================
Logistic Loss
================

.. include::  ../../../includes/objective-function/logloss-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _logloss_c_math:

Computing
---------

Algorithm takes dataset :math:`X = \{ x_1, \ldots, x_n \}` with :math:`n` feature vectors of dimension :math:`p`, vector with correct class labels
:math:`y = \{ y_1, \ldots, y_n \}` and coefficients vector :math:`w = \{ w_0, \ldots, w_p \}` of size :math:`p + 1` as input. Then it calculates
logistic loss, its gradient or gradient using the following formulas.

#####
Value
#####

:math:`L(X, w, y) = \sum_{i = 1}^{n} -y_i \log(prob_i) - (1 - y_i) \log(prob_i)`, where :math:`prob_i = \sigma(w_0 + \sum_{j=1}^{p} w_j x_{i, j})` - predicted probabilities,
:math:`\sigma(x) = \frac{1}{1 + \exp(-x)}` - sigmoid function. Note that probabilities are binded to interval :math:`[\epsilon, 1 - \epsilon]` to avoid problems with
computing log function (:math:`\epsilon=10^{-7}` if float type is used and :math:`10^{-15}` otherwise)

########
Gradient
########

:math:`\overline{grad} = \frac{\partial L}{\partial w}`, where :math:`\overline{grad}_0 = \sum_{i=1}^{n} prob_i - y_i`,
:math:`\overline{grad}_j = \sum_{i=1}^n X_{i, j} (prob_i - y_i) + L1 \cdot |w_j| + 2 \cdot L2 w_j` for :math:`1 \leq j \leq p`

#######
Hessian
#######

:math:`H = (h_{ij}) = \frac{\partial L}{\partial w \partial w}`, where
:math:`h_{0,0}= \sum_{k=1}^n prob_k (1 - prob_k)`,
:math:`h_{i,0} = h_{0,i} = \sum_{k=1}^n X_{k,i} \cdot prob_k (1 - prob_k) ` for :math:`1 \leq i \leq p`,
:math:`h_{i,j} = \sum_{k=1}^n X_{k,i} X_{k,j} \cdot prob_k (1 - prob_k) + [i = j] 2 \cdot L2` for :math:`1 \leq i, j \leq p`


.. _logloss_c_dense_batch:

Computation method: *dense_batch*
---------------------------------
The method computes value of objective function, its gradient or hessian for the dense data.
This is the default and the only method supported.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: LogisticLoss <api_logloss>`.

----------------
Distributed mode
----------------

Currently algorithm does not support distributed execution in SMPD mode.

