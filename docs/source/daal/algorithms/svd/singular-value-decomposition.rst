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

.. _svd:

Singular Value Decomposition
============================

.. toctree::
   :glob:
   :maxdepth: 4


Singular Value Decomposition (SVD) is one of matrix factorization
techniques. It has a broad range of applications including
dimensionality reduction, solving linear inverse problems, and data
fitting.

Details
*******

Given the matrix :math:`X` of size :math:`n \times p`, the problem is to compute the
Singular Value Decomposition (SVD) :math:`X = U \Sigma V^t`, where:

- :math:`U` is an orthogonal matrix of size :math:`n \times n`

- :math:`\Sigma` is a rectangular diagonal matrix of size :math:`n \times p` with non-negative values on the diagonal, called singular values

- :math:`V_t` is an orthogonal matrix of size :math:`p \times p`

Columns of the matrices :math:`U` and :math:`V` are called left and right singular vectors, respectively.

Computation
***********

The following computation modes are available:

.. toctree::
   :maxdepth: 1

   computation-batch-online.rst
   computation-distributed.rst

Examples
********

.. tabs::

   .. tab:: C++ (CPU)

      Batch Processing:

      - :cpp_example:`svd_dense_batch.cpp <svd/svd_dense_batch.cpp>`

      Online Processing:

      - :cpp_example:`svd_dense_online.cpp <svd/svd_dense_online.cpp>`

      Distributed Processing:

      - :cpp_example:`svd_dense_distr.cpp <svd/svd_dense_distr.cpp>`

   .. tab:: Python*

      Batch Processing:

      - :daal4py_example:`svd.py`

      Online Processing:

      - :daal4py_example:`svd_streaming.py`

      Distributed Processing:

      - :daal4py_example:`svd_spmd.py`

Performance Considerations
**************************

To get the best overall performance of singular value decomposition
(SVD), for input, output, and auxiliary data, use homogeneous numeric
tables of the same type as specified in the algorithmFPType class
template parameter.

Online Processing
-----------------

SVD in the online processing mode is at least as computationally
complex as in the batch processing mode and has high memory
requirements for storing auxiliary data between calls to the
compute() method. On the other hand, the online version of SVD may
enable you to hide the latency of reading data from a slow data
source. To do this, implement load prefetching of the next data
block in parallel with the compute() method for the current block.

Online processing mostly benefits SVD when the matrix of left
singular vectors is not required. In this case, memory
requirements for storing auxiliary data goes down from :math:`O(p \cdot n)` to :math:`O(p \cdot p \cdot \text{nblocks})`.

Distributed Processing
----------------------

Using SVD in the distributed processing mode requires gathering local-node :math:`p \times p` numeric tables on the master node.
When the amount of local-node work is small, that is, when the local-node data set is small,
the network data transfer may become a bottleneck.
To avoid this situation, ensure that local nodes have a sufficient amount of work.
For example, distribute input data set across a smaller number of nodes.

.. include:: ../../../opt-notice.rst