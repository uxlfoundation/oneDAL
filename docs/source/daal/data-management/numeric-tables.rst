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

Numeric Tables
==============

.. toctree::
   :maxdepth: 1

   generic-interfaces.rst
   essential-interfaces.rst
   numeric-tables-types.rst

Effective data management is one of the key components for achieving good performance
in data analytics applications. |product| defines the ``NumericTable`` class
that is responsible for storage of and access to the datasets represented
in numeric format on the computational node:

- ``NumericTable`` does not track data available on other nodes.
  The logic that controls synchronization of data between nodes
  should be implemented on the application level.
- ``NumericTable`` does not accumulate information about data
  coming in streaming way. All necessary computations are done
  on the level of the |product| algorithm and/or application software.

The library supports the following data layouts:

-  Heterogeneous, Array Of Structures (AOS)
-  Heterogeneous, Structure Of Arrays (SOA)
-  Homogeneous, dense
-  Homogeneous matrix, dense
-  Homogeneous symmetric matrix, packed
-  Homogeneous triangular matrix, packed
-  Homogeneous, sparse CSR

The optimal data layout for homogeneous and heterogeneous numeric
tables highly depends on a particular algorithm. You can find
algorithm-specific guidance in the Performance Considerations section
for the appropriate algorithm.
