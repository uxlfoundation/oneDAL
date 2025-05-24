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

Basic statistics algorithm computes the following set of quantitative dataset characteristics:

- minimums/maximums
- sums
- means
- sums of squares
- sums of squared differences from the means
- second order raw moments
- variances
- standard deviations
- variations

.. |c_math| replace::   :ref:`Computing <basic_statistics_c_math>`
.. |c_dense| replace::  :ref:`dense <basic_statistics_c_math_dense>`
.. |c_input| replace::  :ref:`compute_input <basic_statistics_c_api_input>`
.. |c_result| replace:: :ref:`compute_result <basic_statistics_c_api_result>`
.. |c_op| replace::     :ref:`compute(...) <basic_statistics_c_api>`

.. |p_math| replace::   :ref:`Partial Computing <basic_statistics_p_math>`
.. |p_input| replace::  :ref:`partial_compute_input <basic_statistics_p_api_input>`
.. |p_result| replace:: :ref:`partial_compute_result <basic_statistics_p_api_result>`
.. |p_op| replace::     :ref:`partial_compute(...) <basic_statistics_p_api>`

.. |f_math| replace::   :ref:`Finalize Computing <basic_statistics_f_math>`
.. |f_op| replace::     :ref:`finalize_compute(...) <basic_statistics_f_api>`

=============  ==========================  ======== =========== ============
**Operation**  **Computational  methods**     **Programming  Interface**
-------------  --------------------------  ---------------------------------
  |c_math|             |c_dense|            |c_op|   |c_input|    |c_result|
  |p_math|             |c_dense|            |p_op|   |p_input|    |p_result|
  |f_math|             |c_dense|            |f_op|   |p_result|   |c_result|
=============  ==========================  ======== =========== ============
