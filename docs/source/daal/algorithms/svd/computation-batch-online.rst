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

Batch and Online Processing
===========================

Online processing computation mode assumes that the data arrives in blocks :math:`i = 1, 2, 3, \ldots \text{nblocks}`.

Algorithm Input
***************

The SVD algorithm accepts the input described below.
Pass the ``Input ID`` as a parameter to the methods that provide input for your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Input for Singular Value Decomposition (Batch and Online Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60

   * - Input ID
     - Input
   * - ``data``
     - Pointer to the numeric table that represents:

       - For batch processing, the entire :math:`n \times p` matrix :math:`X` to be factorized.
       - For online processing, the :math:`n_i \times p` submatrix of :math:`X` that represents
         the current data block in the online processing mode.

       The input can be an object of any class derived from ``NumericTable``.


Algorithm Parameters
********************

The SVD algorithm has the following parameters:

.. tabularcolumns::  |\Y{0.2}|\Y{0.2}|\Y{0.6}|

.. list-table:: Algorithm Parameters for Singular Value Decomposition (Batch and Online Processing)
   :header-rows: 1
   :align: left
   :widths: 10 20 30
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
   * - ``leftSingularMatrix``
     - ``requiredInPackedForm``
     - Specifies whether the matrix of left singular vectors is required. Can be:

       - ``notRequired`` - the matrix is not required
       - ``requiredInPackedForm`` - the matrix in the packed format is required

   * - ``rightSingularMatrix``
     - ``requiredInPackedForm``
     - Specifies whether the matrix of left singular vectors is required. Can be:

       - ``notRequired`` - the matrix is not required
       - ``requiredInPackedForm`` - the matrix in the packed format is required

Algorithm Output
****************

The SVD algorithm calculates the results described below.
Pass the ``Result ID`` as a parameter to the methods that access the results of your algorithm.

.. tabularcolumns::  |\Y{0.2}|\Y{0.8}|

.. list-table:: Algorithm Output for Singular Value Decomposition (Batch and Online Processing)
   :header-rows: 1
   :align: left
   :widths: 10 60
   :class: longtable

   * - Result ID
     - Result
   * - ``singularValues``
     - Pointer to the :math:`1 \times p` numeric table with singular values (the diagonal of the matrix :math:`\Sigma`).
   * - ``leftSingularMatrix``
     - Pointer to the :math:`n \times p` numeric table with left singular vectors (matrix :math:`U`).
       Pass ``NULL`` if left singular vectors are not required.
   * - ``rightSingularMatrix``
     - Pointer to the :math:`p \times p` numeric table with right singular vectors (matrix :math:`V`).
       Pass ``NULL`` if right singular vectors are not required.

.. note::
    By default, these results are objects of the ``HomogenNumericTable`` class,
    but you can define the result as an object of any class derived from ``NumericTable``
    except ``PackedSymmetricMatrix``, ``PackedTriangularMatrix``, and ``CSRNumericTable``.
