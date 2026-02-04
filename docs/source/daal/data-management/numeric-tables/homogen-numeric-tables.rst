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
``HomogenNumericTable`` class when all the features are of the same basic data type.
Values of the features are laid out in memory as one contiguous block in the
row-major order, that is, *Observation 1*, *Observation 2*, and so on.
