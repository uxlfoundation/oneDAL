.. Copyright 2023 Intel Corporation
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

==================
Objective function
==================

.. _alg_objective_function:


.. include::  ../../../includes/objective-function/objective-function-introduction.rst


-----------------------------
Supported objective functions
-----------------------------

.. toctree::
   :titlesonly:

   logloss.rst

------------------------
Mathematical formulation
------------------------

.. _objective_function_c_math:

Computing
---------

Algorithm takes dataset :math:`X = \{ x_1, \ldots, x_n \}` with :math:`n` feature vectors of dimension :math:`p`, vector with correct class labels
:math:`y = \{ y_1, \ldots, y_n \}` and coefficients vector `w = \{ w_0, \ldots, w_p \}`of size :math:`p + 1` as input. Then it calculates
logistic loss, its gradient or hessian.

#####
Value
#####

:math:`L(X, w, y)` - value of objective function.

########
Gradient
########

:math:`\overline{grad} = \frac{\partial L}{\partial w}` - gradient of objective function.

#######
Hessian
#######

:math:`H = (h_{ij}) = \frac{\partial L}{\partial w \partial w}` - hessian of objective function.

.. _objective_function_c_dense_batch:

Computation method: *dense_batch*
---------------------------------
The method computes value of objective function, its gradient or hessian for the dense data.
This is the default and the only method supported.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: Objective Function <api_objective_function>`.

----------------
Distributed mode
----------------

Currently algorithm does not support distributed execution in SMPD mode.

.. rubric:: Examples: Logistic Loss

.. include::  ../../../includes/objective-function/logloss-examples.rst



