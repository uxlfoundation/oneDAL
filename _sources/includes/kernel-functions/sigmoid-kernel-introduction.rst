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

The Sigmoid kernel is a popular kernel function
used in kernelized learning algorithms.

.. |c_math| replace::   :ref:`dense <sigmoid_kernel_c_math>`
.. |c_dense| replace::  :ref:`dense <sigmoid_kernel_c_math_dense>`
.. |c_input| replace::  :ref:`compute_input <sigmoid_kernel_c_api_input>`
.. |c_result| replace:: :ref:`compute_result <sigmoid_kernel_c_api_result>`
.. |c_op| replace::     :ref:`compute(...) <sigmoid_kernel_c_api>`

=============  ===============  =========  =============  ===========
**Operation**  **Computational  methods**  **Programming  Interface**
-------------  --------------------------  --------------------------
|c_math|       |c_dense|        |c_op|     |c_input|      |c_result|
=============  ===============  =========  =============  ===========
