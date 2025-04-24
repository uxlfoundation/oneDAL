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

.. _cor_cov:

Correlation and Variance-Covariance Matrices
============================================

.. toctree::
   :glob:
   :maxdepth: 4

Variance-covariance and correlation matrices are among the most
important quantitative measures of a data set that characterize
statistical relationships involving dependence.

Specifically, the covariance measures the extent to which variables
"fluctuate together" (that is, co-vary). The correlation is the
covariance normalized to be between -1 and +1. A positive correlation
indicates the extent to which variables increase or decrease
simultaneously. A negative correlation indicates the extent to which
one variable increases while the other one decreases. Values close to
+1 and -1 indicate a high degree of linear dependence between
variables.


Details
*******

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})` of
dimension :math:`p`, the problem is to compute the sample means and
variance-covariance matrix or correlation matrix:

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Correlation and Variance-Covariance Matrices
   :widths: 10 60
   :header-rows: 1
   :class: longtable

   * - Statistic
     - Definition
   * - Means
     - :math:`M = (m(1), \ldots , m(p))`, where :math:`m\left(j\right)=\frac{1}{n}\sum _{i}{x}_{ij}`
   * - Variance-covariance matrix
     - :math:`Cov = (v_{ij})`, where :math:`v_{ij}=\frac{1}{n-1}\sum_{k=1}^{n}(x_{ki}-m(i))(x_{kj}-m(j))`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`
   * - Correlation matrix
     - :math:`Cor = (c_{ij})`, where :math:`c_{ij}=\frac{v_{ij}}{\sqrt{v_{ii}\cdot v_{jj}}}`, :math:`i=\overline{1,p}`, :math:`j=\overline{1,p}`

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

  .. tab:: C++ (CPU)

    Batch Processing:

    -  :cpp_example:`cov_dense_batch.cpp <covariance/cov_dense_batch.cpp>`
    -  :cpp_example:`cov_csr_batch.cpp <covariance/cov_csr_batch.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`covariance.py`

    Online Processing:

    - :daal4py_example:`covariance_streaming.py`

    Distributed Processing:

    - :daal4py_example:`covariance_spmd.py`

Performance Considerations
**************************

To get the best overall performance when computing correlation or
variance-covariance matrices:

-  If input data is homogeneous, provide the input data and store
   results in homogeneous numeric tables of the same type as
   specified in the algorithmFPType class template parameter.
-  If input data is non-homogeneous, use AOS layout rather than SOA
   layout.


.. include:: ../../../opt-notice.rst
