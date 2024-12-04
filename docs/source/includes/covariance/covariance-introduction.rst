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

In statistics, covariance and correlation are two of the most fundamental measures of linear dependence between two random variables.
The covariance and the correlation represent the joint variability of any two features.
The correlation is dimensionless, while the covariance is measured in units obtained by multiplying the units of the two features.
Another important distinction is that covariance can be affected by the higher variance of one feature, while
correlation removes the effect of the variances by normalizing the covariance of two features by their square-root of variances.
Their usage is application-dependent. The covariance algorithm computes the following:

- Means
- Covariance (sample and estimated by maximum likelihood method)
- Correlation

.. |c_math| replace::   :ref:`Computing <covariance_c_math>`
.. |p_math| replace::   :ref:`Partial Computing <covariance_p_math>`
.. |f_math| replace::   :ref:`Finalize Computing <covariance_f_math>`
.. |c_dense| replace::  :ref:`dense <covariance_c_math_dense>`
.. |c_input| replace::  :ref:`compute_input <covariance_c_api_input>`
.. |c_result| replace:: :ref:`compute_result <covariance_c_api_result>`
.. |c_op| replace::     :ref:`compute(...) <covariance_c_api>`
.. |p_input| replace::  :ref:`partial_compute_input <covariance_p_api_input>`
.. |p_result| replace:: :ref:`partial_compute_result <covariance_p_api_result>`
.. |p_op| replace::     :ref:`partial_compute(...) <covariance_p_api>`
.. |f_op| replace::     :ref:`finalize_compute(...) <covariance_f_api>`

=============  ==========================  ======== ============ ===========
**Operation**  **Computational  methods**     **Programming  Interface**
-------------  --------------------------  ---------------------------------
  |c_math|             |c_dense|            |c_op|   |c_input|    |c_result|
  |p_math|             |c_dense|            |p_op|   |p_input|    |p_result|
  |f_math|             |c_dense|            |f_op|   |p_result|   |c_result|
=============  ==========================  ======== ============ ===========
