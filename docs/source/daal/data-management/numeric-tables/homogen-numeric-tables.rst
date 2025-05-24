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

.. _homogeneous_numeric_table:

Homogeneous Numeric Tables
==========================

.. toctree::
   :maxdepth: 1
   :hidden:

Use homogeneous numeric tables, that is, objects of the
``HomogenNumericTable`` class, and matrices, that is, objects of the
``Matrix``, ``PackedTriangularMatrix``, and ``PackedSymmetricMatrix`` classes,
when all the features are of the same basic data type. Values of the
features are laid out in memory as one contiguous block in the
row-major order, that is, *Observation 1*, *Observation 2*, and so on. In
|product|, ``Matrix`` is a homogeneous numeric table most suitable for
matrix algebra operations.

For triangular and symmetric matrices with reduced memory footprint,
special classes are available: ``PackedTriangularMatrix`` and
``PackedSymmetricMatrix``. Use the DataLayout enumeration to choose
between representations of triangular and symmetric matrices:

-  Lower packed: ``lowerPackedSymetricMatrix`` or ``lowerPackedTriangularMatrix``
-  Upper packed: ``upperPackedTriangularMatrix`` or ``upperPackedSymetricMatrix``

.. figure:: ./images/packed-storage-format.png
  :width: 600
  :alt:

  Packed Storage Format for Symmetric and Triangular Matrices
