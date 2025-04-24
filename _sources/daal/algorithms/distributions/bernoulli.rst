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

Bernoulli Distribution
======================

Generates Bernoulli distributed random numbers.

Details
*******

Bernoulli random number generator fills the :math:`n \times p` numeric table with
Bernoulli distributed values with the :math:`p` probability of success on a single trial, where :math:`p \in R`,
:math:`0 \leq p \leq 1`.

A variate is called Bernoulli distributed if after a trial it is equal to :math:`1`
with the probability of success :math:`p` and to :math:`0` with the probability :math:`1 - p`.
The probability distribution is given by:

.. math::
    p\{x = 1\} = p

.. math::
    p\{x = 0\} = 1 - p

The cumulative distribution function is as follows:

.. math::
    F_p(x) =
    \begin{cases}
      0, & x < 0 \\
		  1 - p, & 0 \leq x < 1, x \in \mathbb{R} \\
      1, & x \geq 1
    \end{cases}

Batch Processing
****************

.. rubric:: Algorithm Parameters

Bernoulli distribution algorithm has the following parameters in addition to the common parameters specified in :ref:`distributions`:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Bernoulli Distribution (Batch Processing)
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
   * - ``p``
     - Not applicable
     - Success probability of a trial, required parameter.

Examples
********

.. tabs::

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`distributions_bernoulli.py`

Performance Considerations
**************************

To get the best overall performance when using the Bernoulli distribution random number generator,
provide the 32-bit signed integer homogeneous numeric table constructed with enabled equal features.

.. include:: ../../../opt-notice.rst
