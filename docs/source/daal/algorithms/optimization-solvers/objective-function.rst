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

.. _objective_function:

Objective Function
==================

In |short_name|, the objective function represents an interface of objective functions :math:`K(\theta) = F(\theta) + M(\theta)`,
where :math:`F(\theta)` is a smooth and :math:`M(\theta)` is a non-smooth functions, that accepts input argument
:math:`\theta \in R^{p}` and returns:

- The value of objective function, :math:`y = K(\theta)`
- The value of :math:`M(\theta)`, :math:`y_{ns} = M(\theta)`
- The gradient of :math:`F(\theta)`:

  .. math::

    g(\theta) = \nabla F(\theta) = \{ \frac{\partial F}{\partial \theta_1}, \ldots, \frac{\partial F}{\partial \theta_p} \}

- The Hessian of :math:`F(\theta)`:

  .. math::
    H =  = \nabla^2 F(\theta) =
	{\nabla }^{2}{F}_{i}=\left[\begin{array}{ccc}\frac{\partial {F}_{i}}
    {\partial {\theta }_{1}\partial {\theta }_{1}}& \cdots & \frac{\partial {F}_{i}}
    {\partial {\theta }_{1}\partial {\theta }_{p}}\\ ⋮& \ddots & ⋮\\
    \frac{\partial {F}_{i}}{\partial p\partial {\theta }_{1}}& \cdots &
    \frac{\partial {F}_{i}}{\partial {\theta }_{p}\partial {\theta }_{p}}\end{array}\right]

- The objective function specific projection of proximal operator (see [MSE, Log-Loss, Cross-Entropy] for details):

  .. math::

    \text{prox}_{\eta}^{M} (x) = \text{argmin}_{u \in R^p} (M(u) + \frac{1}{2 \eta} |u - x|_2^2)

    x \in R^p

- The objective function specific Lipschitz constant, :math:`\text{constantOfLipschitz} \leq |\nabla| F(\theta)`.

.. toctree::
   :maxdepth: 1
   :caption: Objective functions

   objective-functions/computation.rst
   objective-functions/sum-of-functions.rst
   objective-functions/mse.rst
   objective-functions/with-precomputed-characteristics.rst
   objective-functions/logistic-loss.rst
   objective-functions/cross-entropy.rst

.. note:: On GPU, only :ref:`logistic_loss` and :ref:`cross_entropy_loss` are supported, :ref:`mse` is not supported.