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

Optimization Solvers
====================

An optimization solver is an algorithm to solve an optimization problem, that is,
to find the maximum or minimum of an objective function in the presence of constraints on its variables.
In |short_name| the optimization solver represents the interface of algorithms that search
for the argument :math:`\theta_{*}` that minimizes the function :math:`K(\theta)`:

.. math::

    \theta_{*} = \text{argmin}_{\theta \in \Theta} K(\theta)


.. toctree::
   :maxdepth: 2

   objective-function.rst
   iterative-solver.rst
