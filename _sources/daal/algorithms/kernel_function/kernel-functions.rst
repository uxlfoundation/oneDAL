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

.. _kernel:

Kernel Functions
----------------

.. note::

   Kernel functions are also available with oneAPI interfaces:

   - :ref:`alg_linear_kernel`
   - :ref:`alg_rbf_kernel`

.. toctree::
   :glob:
   :maxdepth: 4

Kernel functions form a class of algorithms for pattern analysis. The
main characteristic of kernel functions is a distinct approach to
this problem. Instead of reducing the dimension of the original data,
kernel functions map the data into higher-dimensional spaces in order
to make the data more easily separable there.


Linear Kernel
=============

A linear kernel is the simplest kernel function.

Problem Statement
*****************

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, x_{mp})`, the problem is to
compute the linear kernel function :math:`K(x_i,, y_i)` for
any pair of input vectors: :math:`K(x_i, y_i) = k {X_i}^T y_i + b`.


Batch Processing
****************

Algorithm Input
+++++++++++++++

The linear kernel function accepts the input described below. Pass
the ``Input ID`` as a parameter to the methods that provide input for
your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Linear Kernel (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60
   :class: longtable

   * - Input ID
     - Input
   * - X
     - Pointer to the :math:`n \times p` numeric table that represents the matrix X. This table can be an object of any class derived from NumericTable.
   * - Y
     - Pointer to the :math:`m \times p` numeric table that represents the matrix Y. This table can be an object of any class derived from NumericTable.

Algorithm Parameters
++++++++++++++++++++

The linear kernel function has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Linear Kernel (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 10 60
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Available computation methods:

       + ``defaultDense`` - default performance-oriented method
       + ``fastCSR`` - performance-oriented method for CSR numeric tables

   * - ``computationMode``
     - ``matrixMatrix``
     - Computation mode for the kernel function. Can be:

       For CPU:

       + ``vectorVector`` - compute the kernel function for given feature vectors :math:`x_i` and :math:`y_j`
       + ``matrixVector`` - compute the kernel function for all vectors in the set :math:`X` and a given feature vector :math:`y_j`
       + ``matrixMatrix`` - compute the kernel function for all vectors in the sets :math:`X` and :math:`Y`.
         In |product|, this mode requires equal numbers of observations in both input tables: :math:`n = m`.

       For GPU:

       + ``matrixMatrix`` - compute the kernel function for all vectors in the sets :math:`X` and :math:`Y`.
         In |product|, this mode requires equal numbers of observations in both input tables: :math:`n = m`.

   * - ``rowIndexX``
     - :math:`0`
     - Index i of the vector in the set :math:`X` for the ``vectorVector`` computation mode.
   * - ``rowIndexY``
     - :math:`0`
     - Index :math:`j` of the vector in the set :math:`Y` for the ``vectorVector`` or ``matrixVector`` computation mode.
   * - ``rowIndexResult``
     - :math:`0`
     - Row index in the values numeric table to locate the result of the computation for the ``vectorVector`` computation mode.
   * - :math:`k`
     - :math:`1.0`
     - The coefficient :math:`k` of the linear kernel.
   * - :math:`b`
     - :math:`0.0`
     - The coefficient :math:`b` of the linear kernel.

Algorithm Output
++++++++++++++++

The linear kernel function calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Linear Kernel (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60

   * - Result ID
     - Result
   * - ``values``
     - Pointer to the :math:`n \times m` numeric table with the values of the kernel
       function.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.

Examples
++++++++

.. tabs::

  .. tab:: oneAPI DPC++

    Batch Processing:

    - :ref:`dpc_linear_kernel_dense_batch.cpp`

  .. tab:: oneAPI C++

    Batch Processing:

    - :ref:`cpp_linear_kernel_dense_batch.cpp`

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`kernel_func_lin_dense_batch.cpp <kernel_function/kernel_func_lin_dense_batch.cpp>`
    - :cpp_example:`kernel_func_lin_csr_batch.cpp <kernel_function/kernel_func_lin_csr_batch.cpp>`

.. Python*:

.. - kernel_func_lin_dense_batch.py
.. - kernel_func_lin_csr_batch.py


Radial Basis Function Kernel
============================

The Radial Basis Function (RBF) kernel is a popular kernel function
used in kernelized learning algorithms.

Problem Statement
*****************

Given a set :math:`X` of :math:`n` feature vectors :math:`x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})`
of dimension :math:`p` and a set :math:`Y` of :math:`m`
feature vectors :math:`y_1 = (y_{11}, \ldots, y_{1p}), \ldots, y_m = (y_{m1}, \ldots, x_{mp})`,
the problem is to compute the RBF kernel function :math:`K(x_i,, y_i)` for any pair of input vectors:

.. math::
   K\left({x}_{i},{y}_{j}\right)=exp\left(-\frac{{\left(\|{x}_{i}-{y}_{j}\|\right)}^{2}}{2{\sigma }^{2}}\right)

Batch Processing
****************

Algorithm Input
+++++++++++++++

The RBF kernel accepts the input described below. Pass the Input
ID as a parameter to the methods that provide input for your
algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Radial Basis Function Kernel (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60
   :class: longtable

   * - Input ID
     - Input
   * - :math:`X`
     - Pointer to the :math:`n \times p` numeric table that represents the matrix :math:`X`. This table can be an object of any class derived from ``NumericTable``.
   * - :math:`Y`
     - Pointer to the :math:`m \times p` numeric table that represents the matrix :math:`Y`. This table can be an object of any class derived from ``NumericTable``.

Algorithm Parameters
++++++++++++++++++++

The RBF kernel has the following parameters:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Radial Basis Function Kernel (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 10 60
   :class: longtable

   * - Parameter
     - Default Value
     - Description
   * - ``algorithmFPType``
     - ``float``
     - The floating-point type that the algorithm uses for intermediate computations. Can be ``float`` or ``double``.
   * - ``method``
     - ``defaultDense``
     - Available computation methods:

       + ``defaultDense`` - default performance-oriented method
       + ``fastCSR`` - performance-oriented method for CSR numeric tables

   * - ``computationMode``
     - ``matrixMatrix``
     - Computation mode for the kernel function. Can be:

       For CPU:

       + ``vectorVector`` - compute the kernel function for given feature vectors :math:`x_i` and :math:`y_j`
       + ``matrixVector`` - compute the kernel function for all vectors in the set :math:`X` and a given feature vector :math:`y_j`
       + ``matrixMatrix`` - compute the kernel function for all vectors in the sets :math:`X` and :math:`Y`.
         In |product|, this mode requires equal numbers of observations in both input tables: :math:`n = m`.

       For GPU:

       + ``matrixMatrix`` - compute the kernel function for all vectors in the sets :math:`X` and :math:`Y`.
         In |product|, this mode requires equal numbers of observations in both input tables: :math:`n = m`.

   * - ``rowIndexX``
     - :math:`0`
     - Index :math:`i` of the vector in the set :math:`X` for the ``vectorVector`` computation mode.
   * - ``rowIndexY``
     - :math:`0`
     - Index :math:`j` of the vector in the set :math:`Y` for the ``vectorVector`` or ``matrixVector`` computation mode.
   * - ``rowIndexResult``
     - :math:`0`
     - Row index in the values numeric table to locate the result of the computation for the ``vectorVector`` computation mode.
   * - ``sigma``
     - :math:`1.0`
     - The coefficient :math:`\sigma` of the RBF kernel.

Algorithm Output
++++++++++++++++

The RBF kernel calculates the results described below. Pass the
Result ID as a parameter to the methods that access the results of
your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Radial Basis Function Kernel (Batch Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60

   * - Result ID
     - Result
   * - ``values``
     - Pointer to the :math:`n \times m` numeric table with the values of the kernel
       function.

       .. note::

          By default, this result is an object of the ``HomogenNumericTable`` class,
          but you can define the result as an object of any class derived from ``NumericTable``
          except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.


Examples
********

.. tabs::

  .. tab:: oneAPI DPC++

    Batch Processing:

    - :ref:`dpc_rbf_kernel_dense_batch.cpp`

  .. tab:: oneAPI C++

    Batch Processing:

    - :ref:`cpp_rbf_kernel_dense_batch.cpp`

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`kernel_func_rbf_dense_batch.cpp <kernel_function/kernel_func_rbf_dense_batch.cpp>`
    - :cpp_example:`kernel_func_rbf_csr_batch.cpp <kernel_function/kernel_func_rbf_csr_batch.cpp>`

.. Python*:

.. - kernel_func_rbf_dense_batch.py
.. - kernel_func_rbf_csr_batch.py
