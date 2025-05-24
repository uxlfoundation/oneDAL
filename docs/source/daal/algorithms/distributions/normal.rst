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

Normal Distribution
===================

Generates normally distributed random numbers.

Details
*******

Normal (Gaussian) random number generator fills the input n x p numeric table with Gaussian random numbers with mean α and standard deviation σ, where α, σ∈R and σ > 0. The probability density function is given by:

.. math::
    f_{\alpha, \sigma}(x) = \frac {1}{\sigma \sqrt{2\pi}}\exp(-\frac {(x - a)^2}{2\sigma^2}), -\infty < x < +\infty

The cumulative distribution function is as follows:

.. math::
    F_{\alpha, \sigma}(x) = \int _{-\infty}^{x} \frac {1}{\sigma \sqrt{2\pi}} \exp(-\frac {(y - a)^2}{2\sigma^2})dy, -\infty < x < +\infty

Batch Processing
****************

.. rubric:: Algorithm Parameters

Normal distribution algorithm has the following parameters in addition to the common parameters specified in :ref:`distributions`:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Normal Distribution (Batch Processing)
   :header-rows: 1
   :widths: 10 10 60
   :align: left
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
       The only method supported so far is the Inverse Cumulative Distribution Function (ICDF) method.
   * - ``a``
     - :math:`0`
     - The mean :math:`\alpha`
   * - ``sigma``
     - :math:`1`
     - The standard deviation :math:`\sigma`

Examples
********

.. tabs::

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`distributions_normal.py`
