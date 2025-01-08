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

QR Decomposition
================

QR decomposition is a matrix factorization technique that decomposes a matrix into
a product of an orthogonal matrix :math:`Q` and an upper triangular matrix :math:`R`.

QR decomposition is used in solving linear inverse and least squares problems.
It also serves as a basis for algorithms that find eigenvalues and eigenvectors.

.. toctree::
   :maxdepth: 1

   qr-without-pivoting.rst
   qr-pivoted.rst

Performance Considerations
**************************

To get the best overall performance of the QR decomposition, for input, output, and auxiliary data,
use homogeneous numeric tables of the same type as specified in the ``algorithmFPType`` class template parameter.

Online Processing
-----------------

QR decomposition in the online processing mode is at least as computationally complex as in the batch processing mode
and has high memory requirements for storing auxiliary data between calls to the ``compute()s`` method.
On the other hand, the online version of QR decomposition may enable you to hide the latency of reading data from a slow data source.
To do this, implement load prefetching of the next data block in parallel with the ``compute()`` method for the current block.

Online processing mostly benefits QR decomposition when the matrix Q is not required.
In this case, memory requirements for storing auxiliary data goes down from
:math:`O(p \cdot n)` to :math:`O(p \cdot p \cdot \text{nblocks})`.

Distributed Processing
----------------------

Using QR decomposition in the distributed processing mode requires gathering local-node :math:`p \times p` numeric tables on the master node.
When the amount of local-node work is small, that is, when the local-node data set is small,
the network data transfer may become a bottleneck. To avoid this situation, ensure that local nodes have a sufficient amount of work.
For example, distribute the input data set across a smaller number of nodes.

.. include:: ../../../opt-notice.rst