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

Heterogeneous Numeric Tables
============================

.. toctree::
   :maxdepth: 1
   :hidden:

Heterogeneous numeric tables enable you to deal with data structures
that are of different data types by nature. |product| provides two
ways to represent non-homogeneous numeric tables: AOS and SOA.

.. _aos_layout:

AOS Numeric Table
*****************

AOS Numeric Table provides access to observations (feature vectors)
that are laid out in a contiguous memory block:

.. figure:: ./images/aos-layout.png
  :width: 600
  :alt:

  Array-Of-Structures (AOS) Memory Layout

Examples
--------

C++: :cpp_example:`datasource/datastructures_aos.cpp`

.. Python*: datastructures_aos.py

.. _soa_layout:

SOA Numeric Table
*****************

SOA Numeric Table provides access to data sets where observations
for each feature are laid out contiguously in memory:

.. figure:: ./images/soa-layout.png
  :width: 600
  :alt:

  Structure-Of-Arrays (SOA) Memory Layout

Examples
--------

C++: :cpp_example:`datasource/datastructures_soa.cpp`

.. Python*:datastructures_soa.py
