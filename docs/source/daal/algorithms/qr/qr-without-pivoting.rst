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

QR Decomposition without Pivoting
=================================

Given the matrix :math:`X` of size :math:`n \times p`, the problem is to compute the QR decomposition
:math:`X = QR`, where

- :math:`Q` is an orthogonal matrix of size :math:`n \times n`
- :math:`R` is a rectangular upper triangular matrix of size :math:`n \times p`

The library requires :math:`n > p`. In this case:

.. math::

    X = QR = [Q_1, Q_2] \cdot \begin{bmatrix} R_1 \\ 0 \end{bmatrix} = Q_1 R_1

where the matrix :math:`Q_1` has the size :math:`n \times p` and :math:`R_1` has the size :math:`p \times p`.

Computation
***********

The following computation modes are available:

.. toctree::
   :maxdepth: 1

   without-pivoting/computation-batch-online.rst
   without-pivoting/computation-distributed.rst

Examples
********

.. tabs::

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`qr_dense_batch.cpp <qr/qr_dense_batch.cpp>`

    Online Processing:

    - :cpp_example:`qr_dense_online.cpp <qr/qr_dense_online.cpp>`

    Distributed Processing:

    - :cpp_example:`qr_dense_distr.cpp <qr/qr_dense_distr.cpp>`


  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`qr.py`

    Online Processing:

    - :daal4py_example:`qr_streaming.py`

    Distributed Processing:

    - :daal4py_example:`qr_spmd.py`
