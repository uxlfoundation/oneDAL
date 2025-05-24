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

.. _pca:

Principal Component Analysis
============================

.. note::

   Principal Component Analysis is also available with oneAPI interfaces:

   - :ref:`alg_pca`

.. toctree::
   :glob:
   :maxdepth: 4


Principal Component Analysis (PCA) is a method for exploratory data
analysis. PCA transforms a set of observations of possibly correlated
variables to a new set of uncorrelated variables, called principal
components. Principal components are the directions of the largest
variance, that is, the directions where the data is mostly spread
out.

Because all principal components are orthogonal to each other, there
is no redundant information. This is a way of replacing a group of
variables with a smaller set of new variables. PCA is one of powerful
techniques for dimension reduction.

Details
*******

Given a set :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})\}` of :math:`p`-dimensional
feature vectors or a :math:`p \times p` correlation matrix and the number of principal components :math:`p_r`,
the problem is to compute :math:`p_r` principal directions
(eigenvectors) for the data set. The library returns the
transformation matrix :math:`T` of size :math:`p_r \times p`, which contains
eigenvectors in the row-major order and a vector of respective
eigenvalues in descending order.

|product| provides two methods for running PCA:

-  SVD
-  Correlation

Eigenvectors computed by PCA are not uniquely defined due to sign
ambiguity. PCA supports fast ad-hoc "sign flip" technique described
in the paper [Bro07]_. It modifies the signs of eigenvectors shown below:

.. math::
   \hat{T_i} = T_i \cdot sgn(\max_{1 \leq j \leq p } |{T}_{i,j}|), i=1, \ldots ,p_r

where :math:`T`-transformation matrix is computed by PCA, :math:`T_i` - :math:`i`-th
row in the matrix, :math:`j` - column number, :math:`sgn` - signum function:

.. math::
   sgn(x) =
   \begin{cases}
      -1, & x < 0,\\
         0, & x = 0, \\
      1, & x > 0
   \end{cases}

You can provide these types of input data to the PCA algorithms of
the library:

-  Original, non-normalized data set

-  Normalized data set, where each feature has the zero mean and unit variance

-  Correlation matrix

Computation
***********

The following computation modes are available:

.. toctree::
   :maxdepth: 1

   computation-batch.rst
   computation-online.rst
   computation-distributed.rst

Examples
********

.. tabs::

   .. tab:: oneAPI DPC++

      Batch Processing:

      - :ref:`dpc_pca_cor_dense_batch.cpp`

   .. tab:: oneAPI C++

      Batch Processing:

      - :ref:`cpp_pca_cor_dense_batch.cpp`

   .. tab:: C++ (CPU)

      Batch Processing:

      - :cpp_example:`pca_cor_dense_batch.cpp <pca/pca_cor_dense_batch.cpp>`
      - :cpp_example:`pca_cor_csr_batch.cpp <pca/pca_cor_csr_batch.cpp>`
      - :cpp_example:`pca_svd_dense_batch.cpp <pca/pca_svd_dense_batch.cpp>`

      Online Processing:

      - :cpp_example:`pca_cor_dense_online.cpp <pca/pca_cor_dense_online.cpp>`
      - :cpp_example:`pca_cor_csr_online.cpp <pca/pca_cor_csr_online.cpp>`
      - :cpp_example:`pca_svd_dense_online.cpp <pca/pca_svd_dense_online.cpp>`

      Distributed Processing:

      - :cpp_example:`pca_cor_dense_distr.cpp <pca/pca_cor_dense_distr.cpp>`
      - :cpp_example:`pca_cor_csr_distr.cpp <pca/pca_cor_csr_distr.cpp>`
      - :cpp_example:`pca_svd_dense_distr.cpp <pca/pca_svd_dense_distr.cpp>`

   .. tab:: Python*

      Batch Processing:

      - :daal4py_example:`pca.py`

      Distributed Processing:

      - :daal4py_example:`pca_spmd.py`

Performance Considerations
**************************

To get the best overall performance of the PCA algorithm:

-  If input data is homogeneous, provide the input data and store
   results in homogeneous numeric tables of the same type as
   specified in the algorithmFPType class template parameter.
-  If input data is non-homogeneous, use AOS layout rather than SOA
   layout.

PCA computation using the correlation method involves the correlation
and variance-covariance matrices algorithm. Depending on the method
of this algorithm, the performance of PCA computations may vary. For
sparse data sets, use the methods of this algorithm for sparse data.

Batch Processing
----------------

Because the PCA in the batch processing mode performs
normalization for data passed as Input ID, to achieve the best
performance, normalize the input data set. To inform the algorithm
that the data is normalized, set the normalization flag for the
input numeric table that represents your data set by calling the
setNormalizationFlag() method of the NumericTableIface class.

Because the PCA with the correlation method (defaultDense) in the
batch processing mode is based on the computation of the
correlation matrix, to achieve the best performance, precompute
the correlation matrix. To pass the precomputed correlation matrix
to the algorithm, use correlation as Input ID.

Online Processing
-----------------

PCA with the SVD method (svdDense) in the online processing mode
is at least as computationally complex as in the batch processing
mode and has high memory requirements for storing auxiliary data
between calls to compute(). On the other hand, the online version
of the PCA with the SVD method may enable you to hide the latency
of reading data from a slow data source. To do this, implement
load prefetching of the next data block in parallel with the
compute() method for the current block.

Distributed Processing
----------------------

PCA with the SVD method (svdDense) in the distributed processing
mode requires gathering local-node :math:`p \times p` numeric tables on the
master node. When the amount of local-node work is small, that is,
when the local-node data set is small, the network data transfer
may become a bottleneck. To avoid this situation, ensure that
local nodes have a sufficient amount of work. For example,
distribute the input data set across a smaller number of nodes.

.. include:: ../../../opt-notice.rst