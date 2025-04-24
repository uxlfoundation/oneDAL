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

===================
Computational Modes
===================

.. _batch:

-----
Batch
-----
In the batch processing mode, the algorithm works with the entire data set to produce the final
result. A more complex scenario occurs when the entire data set is not available at the moment
or the data set does not fit into the device memory.

.. _online:

------
Online
------
In the online processing mode, the algorithm processes a data set in blocks streamed into the
device's memory. Partial results are updated incrementally and finalized when the last data block
is processed.

.. _distributed:

-----------
Distributed
-----------
In the distributed processing mode, the algorithm operates on a data set
distributed across several devices (compute nodes). On each node, the algorithm
produces partial results that are later merged into the final result on the main
node.