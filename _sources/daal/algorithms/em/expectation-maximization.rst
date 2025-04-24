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

Expectation-Maximization
************************

.. toctree::
   :glob:
   :maxdepth: 4


Expectation-Maximization (EM) algorithm is an iterative method for
finding the maximum likelihood and maximum a posteriori estimates of
parameters in models that typically depend on hidden variables.

While serving as a clustering technique, EM is also used in
non-linear dimensionality reduction, missing value problems, and
other areas.


Details
=======

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p`, the problem is to find a maximum-likelihood estimate of
the parameters of the underlying distribution when the data is
incomplete or has missing values.


Expectation-Maximization (EM) Algorithm in the General Form
-----------------------------------------------------------

Let :math:`X` be the observed data which has log-likelihood :math:`l(\theta; X)`
depending on the parameters :math:`\theta`. Let :math:`X^m` be the latent or
missing data, so that :math:`T=(X, X^m)` is the complete data with
log-likelihood :math:`l_0(\theta; X)`. The algorithm for solving the
problem in its general form is the following EM algorithm
([Dempster77]_,
[Hastie2009]_):

#. Choose initial values of the parameters :math:`\theta^{(0)}`.

#.

 *Expectation step*: in the :math:`j`-th step, compute :math:`Q(\theta', \theta^{(j)}) = E (l_0(\theta'; T) | X, \theta^{(j)})` as a
 function of the dummy argument :math:`\theta'`.

#.

 *Maximization step*: in the :math:`j`-th step, calculate the new
 estimate :math:`\theta^{(j+1)}` by maximizing :math:`Q(\theta', \theta^{(j)})` over :math:`\theta'`.

#.

 Repeat steps 2 and 3 until convergence.


EM algorithm for the Gaussian Mixture Model
-------------------------------------------

Gaussian Mixture Model (GMM) is a mixture of k p-dimensional
multivariate Gaussian distributions represented as

.. math::
	F\left(x{|\alpha }_{1},\ldots,{\alpha }_{k};{\theta }_{1},\ldots,{\theta }_{k}\right)=\sum _{i=1}^{k}{\alpha }_{i}{\int }_{-\infty }^{x}pd\left(y|{\theta }_{i}\right),

where :math:`\sum _{i=1}^{k}{\alpha_i} = 1` and :math:`\alpha_i \geq 0`.

The :math:`pd(x|\theta_i)` is the probability density function with
parameters :math:`\theta_i = (m_i, \Sigma_i)`, where :math:`m_i`
the vector of means, and :math:`\Sigma_i` is the
variance-covariance matrix. The probability density function for a
:math:`p`-dimensional multivariate Gaussian distribution is defined as
follows:

.. math::
	pd\left(x|{\theta }_{i}\right)=\frac{\mathrm{exp}\left(-\frac{1}{2}{\left(x-{m}_{i}\right)}^{T}\sum _{i}^{-1}\left(x-{m}_{i}\right)\right)}{\sqrt{{\left(2\pi \right)}^{p}|\sum _{i}|}}.

Let :math:`x_{ij} = I\{x_i \text{belongs to j mixture component}\}` be
the indicator function and :math:`\theta = (\alpha_1, \ldots, \alpha_k; \theta_1, \ldots, \theta_k)`.


Computation
+++++++++++

The EM algorithm for GMM includes the following steps:

Define the weights as follows:

.. math::
	{w}_{ij}=\frac{pd\left({x}_{i}|{z}_{ij},{\theta }_{j}\right){\alpha }_{j}}{\sum _{r=1}^{k}pd\left({x}_{i}|{z}_{ir},{\theta }_{r}\right){\alpha }_{r}}

for :math:`i = 1, \ldots, n` and :math:`j=1, \ldots, k`.

#. Choose initial values of the parameters: :math:`{\theta }^{\left(0\right)}=\left({\alpha }_{1}^{\left(0\right)},...,{\alpha }_{k}^{\left(0\right)};{\theta }_{1}^{\left(0\right)},...,{\theta }_{k}^{\left(0\right)}\right)`


#. *Expectation step*: in the :math:`j`-th step, compute the matrix :math:`W = {(w_{ij})}_{nxk}` with the weights :math:`w_{ij}`

#. Maximization step: in the :math:`j`-th step, for all :math:`r=1, \ldots, k` compute:

   a.
     The mixture weights :math:`{\alpha }_{r}^{\left(j+1\right)}=\frac{{n}_{r}}{n}`, where :math:`{n}_{r}=\sum _{i=1}^{n}{w}_{ir}`
     is the "amount" of the feature vectors that are assigned
     to the :math:`r`-th mixture component

   b. Mean estimates :math:`{m}_{r}^{\left(j+1\right)}=\frac{1}{{n}_{r}}\sum _{i=1}^{n}{w}_{ir}{x}_{i}`

   c.
     Covariance estimate :math:`\sum _{r}^{(j+1)}=({\sigma }_{r,hg}^{(j+1)})`
     of size :math:`p \times p` with :math:`\sigma_{r,hg}^{(j+1)}=\frac{1}{n_r}\sum_{l=1}^{n}{w}_{lr}(x_{lh}-m_{r,h}^{(j+1)})(x_{lg}-m_{r,g}^{(j+1)})`


#. Repeat steps 2 and 3 until any of these conditions is met:

   - :math:`|\log({\theta }^{(j+1)}-{\theta }^{(j)})|<\epsilon`, where the likelihood function is:

     :math:`\log(\theta)=\sum_{i=1}^{n}\log(\sum _{j=1}^{k}{pd(x}_{i}|{z}_{j},{\theta }_{j}){\alpha }_{j})`

   - The number of iterations exceeds the predefined level.

Initialization
++++++++++++++

The EM algorithm for GMM requires initialized vector of
weights, vectors of means, and variance-covariance
[Biernacki2003, Maitra2009].

The EM initialization algorithm for GMM includes the following
steps:

#. Perform nTrials starts of the EM algorithm with nIterations
   iterations and start values:

   - Initial means - :math:`k` different random observations from the input data set
   - Initial weights - the values of :math:`1/k`
   - Initial covariance matrices - the covariance of the input data

#. Regard the result of the best EM algorithm in terms of the
   likelihood function values as the result of initialization

Initialization
==============

The EM algorithm for GMM requires initialized vector of weights,
vectors of means, and variance-covariance. Skip the initialization
step if you already calculated initial weights, means, and
covariance matrices.

Batch Processing
----------------

Algorithm Input
+++++++++++++++

The EM for GMM initialization algorithm accepts the input
described below. Pass the ``Input ID`` as a parameter to the methods
that provide input for your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Expectation-Maximization Initialization (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left

   * - Input ID
     - Input
   * - data
     - Pointer to the :math:`n \times p` numeric table with the data to which the EM
       initialization algorithm is applied. The input can be an object of any
       class derived from NumericTable.

Algorithm Parameters
++++++++++++++++++++

The EM for GMM initialization algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Algorithm Parameters for Expectation-Maximization Initialization (Batch Processing)
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
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``nComponents``
     - Not applicable
     - The number of components in the Gaussian Mixture Model, a required parameter.
   * - ``nTrials``
     - :math:`20`
     - The number of starts of the EM algorithm.
   * - ``nIterations``
     - :math:`10`
     - The maximal number of iterations in each start of the EM algorithm.
   * - ``accuracyThreshold``
     - 1.0e-04
     - The threshold for termination of the algorithm.
   * - ``covarianceStorage``
     - ``full``
     - Covariance matrix storage scheme in the Gaussian Mixture Model:

       + ``full`` - covariance matrices are stored as numeric tables of size :math:`p \times p`.
         All elements of the matrix are updated during the processing.
       + ``diagonal`` - covariance matrices are stored as numeric tables of size :math:`1 \times p`.
         Only diagonal elements of the matrix are updated during the
         processing, and the rest are assumed to be zero.

   * - ``engine``
     - `SharePtr< engines:: mt19937:: Batch>()`
     - Pointer to the random number generator engine that is used internally to get the initial means in each EM start.

Algorithm Output
++++++++++++++++

The EM for GMM initialization algorithm calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Expectation-Maximization Initialization (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``weights``
     - Pointer to the :math:`1 \times k` numeric table with mixture weights.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

   * - ``means``
     - Pointer to the :math:`k \times p` numeric table with each row containing the estimate
       of the means for the :math:`i`-th mixture component, where :math:`i=0, 1, \ldots, k-1`.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``covariances``
     - Pointer to the ``DataCollection`` object that contains :math:`k` numeric tables,
       each with the :math:`p \times p` variance-covariance matrix for the :math:`i`-th mixture
       component of size:

       + :math:`p \times p` - for the full covariance matrix storage scheme
       + :math:`1 \times p` - for the diagonal covariance matrix storage scheme

       .. note::

          By default, the collection contains objects of the ``HomogenNumericTable`` class,
          but you can define them as objects of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix`` and ``CSRNumericTable``.

Computation
===========

Batch Processing
----------------

Algorithm Input
+++++++++++++++

The EM for GMM algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Expectation-Maximization Computation (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the :math:`n \times p` numeric table with the data to which the EM
       algorithm is applied. The input can be an object of any class derived
       from ``NumericTable``.
   * - ``inputWeights``
     - Pointer to the :math:`1 \times k` numeric table with initial mixture weights. This input can be an object of any class derived from NumericTable.
   * - ``inputMeans``
     - Pointer to a :math:`k \times p` numeric table. Each row in this table contains the
       initial value of the means for the :math:`i`-th mixture component, where :math:`i = 0, 1, \ldots, k-1`.
       This input can be an object of any class derived from ``NumericTable``.
   * - ``inputCovariances``
     - Pointer to the ``DataCollection`` object that contains :math:`k` numeric tables,
       each with the :math:`p \times p` variance-covariance matrix for the :math:`i`-th mixture component of size:

       + :math:`p \times p` - for the full covariance matrix storage scheme
       + :math:`1 \times p` - for the diagonal covariance matrix storage scheme

       The collection can contain objects of any class derived from NumericTable.

   * - ``inputValues``
     - Pointer to the result of the EM for GMM initialization algorithm. The
       result of initialization contains weights, means, and a collection of
       covariances. You can use this input to set the initial values for the EM
       for GMM algorithm instead of explicitly specifying the weights, means,
       and covariance collection.

Algorithm Parameters
++++++++++++++++++++

The EM for GMM algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Algorithm Parameters for Expectation-Maximization Computation (Batch Processing)
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
     - Performance-oriented computation method, the only method supported by the algorithm.
   * - ``nComponents``
     - Not applicable
     - The number of components in the Gaussian Mixture Model, a required parameter.
   * - ``maxIterations``
     - :math:`10`
     - The maximal number of iterations in the algorithm.
   * - ``accuracyThreshold``
     - 1.0e-04
     - The threshold for termination of the algorithm.
   * - ``covariance``
     - Pointer to an object of the BatchIface class
     - Pointer to the algorithm that computes the covariance matrix.

       .. note::

          By default, the respective |product| algorithm is used,
          implemented in the class derived from ``BatchIface``.
   * - ``regularizationFactor``
     - :math:`0.01`
     - Factor for covariance regularization in the case of ill-conditional data.
   * - ``covarianceStorage``
     - ``full``
     - Covariance matrix storage scheme in the Gaussian Mixture Model:

       + ``full`` - covariance matrices are stored as numeric tables of size :math:`p \times p`.
         All elements of the matrix are updated during the processing.

       + ``diagonal`` - covariance matrices are stored as numeric tables of size :math:`1 \times p`.
         Only diagonal elements of the matrix are updated during the processing, and the rest are assumed to be zero.


Algorithm Output
++++++++++++++++

The EM for GMM algorithm calculates the results described below. Pass
the ``Result ID`` as a parameter to the methods that access the results
of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Expectation-Maximization Computation (Batch Processing)
   :widths: 10 60
   :header-rows: 1
   :align: left
   :class: longtable

   * - Result ID
     - Result
   * - ``weights``
     - Pointer to the :math:`1 \times k` numeric table with mixture weights.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.
   * - ``means``
     - Pointer to the :math:`k \times p` numeric table with each row containing the estimate
       of the means for the :math:`i`-th mixture component, where :math:`i=0, 1, \ldots, k-1`.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix``, ``PackedSymmetricMatrix``, and ``CSRNumericTable``.

   * - ``covariances``
     - Pointer to the DataCollection object that contains :math:`k` numeric tables,
       each with the :math:`p \times p` variance-covariance matrix for the :math:`i`-th mixture component of size:

       + :math:`p \times p` - for the full covariance matrix storage scheme
       + :math:`1 \times p` - for the diagonal covariance matrix storage scheme


       .. note::

          By default, the collection contains objects of the ``HomogenNumericTable`` class,
          but you can define them as objects of any class derived from ``NumericTable``
          except ``PackedTriangularMatrix`` and ``CSRNumericTable``.

   * - ``goalFunction``
     - Pointer to the :math:`1 \times 1` numeric table with the value of the logarithm of
       the likelihood function after the last iteration.

       .. note:: By default, this result is an object of the ``HomogenNumericTable`` class.
   * - ``nIterations``
     - Pointer to the :math:`1 \times 1` numeric table with the number of iterations
       computed after completion of the algorithm.

       .. note:: By default, this result is an object of the ``HomogenNumericTable`` class.

Examples
++++++++

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`em_gmm_dense_batch.cpp <em/em_gmm_dense_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`em_gmm.py`

Performance Considerations
==========================

To get the best overall performance of the expectation-maximization
algorithm at the initialization and computation stages:

-  If input data is homogeneous, provide the input data and store
   results in homogeneous numeric tables of the same type as
   specified in the algorithmFPType class template parameter.
-  If input data is non-homogeneous, use AOS layout rather than SOA layout.

.. include:: ../../../opt-notice.rst

