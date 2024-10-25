.. Copyright 2021 Intel Corporation
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

Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed in [Ester96]_.
It is a density-based clustering non-parametric algorithm: given a set of observations in some space,
it groups together observations that are closely packed together (observations with many nearby neighbors),
marking as outliers observations that lie alone in low-density regions (whose nearest neighbors are too far away).

.. |c_math| replace:: :ref:`Compute <dbscan_c_math>`
.. |c_input| replace:: :ref:`compute_input <dbscan_c_api_input>`
.. |c_result| replace:: :ref:`compute_result <dbscan_c_api_result>`
.. |c_op| replace:: :ref:`compute(...) <dbscan_c_api>`

=============== =========================== ======== =========== ============
 **Operation**  **Computational methods**     **Programming Interface**
--------------- --------------------------- ---------------------------------
   |c_math|          Default method          |c_op|   |c_input|   |c_result|
=============== =========================== ======== =========== ============
