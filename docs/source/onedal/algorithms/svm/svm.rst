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

.. _alg_svm:

======================================================
Support Vector Machine Classifier and Regression (SVM)
======================================================

.. include:: ../../../includes/svm/svm-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _svm_t_math:

Training
--------

Given :math:`n` feature vectors :math:`X=\{x_1=(x_{11},\ldots,x_{1p}),\ldots,
x_n=(x_{n1},\ldots,x_{np})\}` of size :math:`p`,
their non-negative observation weights :math:`W=\{w_1,\ldots,w_n\}`,
and :math:`n` responses :math:`Y=\{y_1,\ldots,y_n\}`,

.. tabs::

  .. group-tab:: Classification

    - :math:`y_i \in \{0, \ldots, M-1\}`, where :math:`M` is the number of classes

  .. group-tab:: Regression

    - :math:`y_i \in \mathbb{R}`

  .. group-tab:: Nu-classification

    - :math:`y_i \in \{0, \ldots, M-1\}`, where :math:`M` is the number of classes

  .. group-tab:: Nu-regression

    - :math:`y_i \in \mathbb{R}`

the problem is to build a Support Vector Machine (SVM) classification, regression,
nu-classification, or nu-regression model.

The SVM model is trained using the Sequential minimal optimization (SMO) method [Boser92]_
for reduced to the solution of the quadratic optimization problem

.. tabs::

  .. group-tab:: Classification

    .. math::
          \underset{\alpha }{\mathrm{min}}\frac{1}{2}{\alpha }^{T}Q\alpha -{e}^{T}\alpha

    with :math:`0 \leq \alpha_i \leq C`, :math:`i = 1, \ldots, n`, :math:`y^T \alpha = 0`,
    where :math:`e` is the vector of ones, :math:`C` is the upper bound of the
    coordinates of the vector :math:`\alpha`, :math:`Q` is a symmetric matrix of size :math:`n \times n`
    with :math:`Q_{ij} = y_i y_j K(x_i, x_j)`, and :math:`K(x,y)` is a kernel function.

  .. group-tab:: Regression

    .. math::
          \underset{\alpha }{\mathrm{min}}\frac{1}{2}{\alpha }^{T}Q\alpha -{s}^{T}\alpha

    with :math:`0 \leq \alpha_i \leq C`, :math:`i = 1, \ldots, 2n`, :math:`z^T \alpha = 0`,
    where :math:`C` is the upper bound of the coordinates of the vector :math:`\alpha`,
    :math:`Q` is a symmetric matrix of size :math:`2n \times 2n`
    with :math:`Q_{ij} = y_i y_j K(x_i, x_j)`, and :math:`K(x,y)` is a kernel function.
    Vectors :math:`s` and :math:`z` for the regression problem are formulated according
    to the following rule:

    .. math::
      \begin{cases}
         z_i = +1, s_i = \epsilon - y_i, & 0 < i \leq n \\
         z_i = -1, s_i = \epsilon + y_i, & n < i \leq 2n
      \end{cases}

    Where :math:`\epsilon` is the error tolerance parameter.

  .. group-tab:: Nu-classification

    .. math::
          \underset{\alpha }{\mathrm{min}}\frac{1}{2}{\alpha }^{T}Q\alpha

    with :math:`0 \leq \alpha_i \leq 1`, :math:`i = 1, \ldots, n`,
    :math:`e^T\alpha = n\nu`, :math:`y^T\alpha = 0`,
    where :math:`e` is the vector of ones, :math:`\nu` is an upper bound on the fraction
    of training errors and a lower bound of the fraction of the support vector,
    :math:`Q` is a symmetric matrix of size :math:`n \times n` with :math:`Q_{ij} = y_i y_j K(x_i, x_j)`,
    and :math:`K(x,y)` is a kernel function.

  .. group-tab:: Nu-regression

    .. math::
          \underset{\alpha }{\mathrm{min}}\frac{1}{2}{\alpha }^{T}Q\alpha + z^T\alpha

    with :math:`0 \leq \alpha_i \leq \frac{C}{n}`, :math:`i = 1, \ldots, 2n`,
    :math:`\sum_{i=1}^{n} \alpha_i - \sum_{i=n+1}^{2n} \alpha_i = 0`,
    :math:`\sum_{i=1}^{2n} \alpha_i = C\nu`, where :math:`C` is the upper bound of the coordinates
    of the vector :math:`\alpha`, :math:`\nu` is an upper bound on the fraction
    of training errors and a lower bound of the fraction of the support vector,
    :math:`Q` is a symmetric matrix of size :math:`2n \times 2n`
    with :math:`Q_{ij} = y_i y_j K(x_i, x_j)`, and :math:`K(x,y)` is a kernel function.
    Vector :math:`z` for the regression problem are formulated according
    to the following rule:

    .. math::
      \begin{cases}
         z_i = y_i, & 0 < i \leq n \\
         z_i = y_{i-n}, & n < i \leq 2n
      \end{cases}

Working subset of α updated on each iteration of the algorithm is
based on the Working Set Selection (WSS) 3 scheme [Fan05]_.
The scheme can be optimized using one of these techniques or both:

- **Cache**:
  the implementation can allocate a predefined amount of memory
  to store intermediate results of the kernel computation.

- **Shrinking**:
  the implementation can try to decrease the amount of kernel
  related computations (see [Joachims99]_).

The solution of the problem defines the separating hyperplane and
corresponding decision function :math:`D(x)= \sum_{k} {y_k \alpha_k K(x_k, x)} + b`,
where only those :math:`x_k` that correspond to non-zero :math:`\alpha_k` appear in the sum,
and :math:`b` is a bias. Each non-zero :math:`\alpha_k` is called a dual
coefficient and the corresponding :math:`x_k` is called a support vector.

.. _svm_t_math_smo:

Training method: *smo*
~~~~~~~~~~~~~~~~~~~~~~
In *smo* training method, all vectors from the training dataset are used for each iteration.

.. _svm_t_math_thunder:

Training method: *thunder*
~~~~~~~~~~~~~~~~~~~~~~~~~~
In *thunder* training method, the algorithm iteratively solves the convex optimization problem
with the linear constraints by selecting the fixed set of active constrains (working set) and
applying Sequential Minimal Optimization (SMO) solver to the selected subproblem.
The description of this method is given in Algorithm [Wen2018]_.

.. _svm_i_math:
.. _svm_i_math_smo:
.. _svm_i_math_thunder:

Inference methods: *smo* and *thunder*
--------------------------------------
*smo* and *thunder* inference methods perform prediction in the same way:

Given the SVM classification or regression model and
:math:`r` feature vectors :math:`x_1, \ldots, x_r`,
the problem is to calculate the signed value of the
decision function :math:`D(x_i)`, :math:`i=1, \ldots, r`. The sign of the
value defines the class of the feature vector, and the absolute
value of the function is a multiple of the distance between the
feature vector and the separating hyperplane.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Support Vector Machine Classifier and Regression <api_svm>`.

--------
Examples
--------

.. include:: ../../../includes/svm/svm-examples.rst
