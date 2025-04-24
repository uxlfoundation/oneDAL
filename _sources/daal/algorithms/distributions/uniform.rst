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

Uniform Distribution
====================

Generates random numbers uniformly distributed on the interval :math:`[a, b)`.

Details
*******

Uniform random number generator fills the input :math:`n \times p` numeric table with values
that are uniformly distributed on the interval :math:`[a, b)`, where :math:`a, b \in \mathbb{r}` and :math:`a <> b`.

The probability density is given by:

.. math::
    f_{a, b}(x) =
    \begin{cases}
      \frac {1}{b - a}, & x \in [a, b)\\
      0, & x \notin [a, b)
    \end{cases}

The cumulative distribution function is as follows:

.. math::
    F_{a, b}(x) =
    \begin{cases} 0, & x < a \\
      \frac {x - a}{b - a}, & a \leq x < b \\
      1, & x \geq b
    \end{cases}

Batch Processing
****************

.. rubric:: Algorithm Parameters

Uniform distribution algorithm has the following parameters in addition to the common parameters specified in :ref:`distributions`:

.. tabularcolumns::  |\Y{0.15}|\Y{0.15}|\Y{0.7}|

.. list-table:: Algorithm Parameters for Uniform Distribution (Batch Processing)
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
   * - ``a``
     - :math:`0.0`
     - The left bound :math:`a`.
   * - ``b``
     - :math:`1.0`
     - The right bound :math:`b`.

Examples
********

.. tabs::

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`distributions_uniform.py`
