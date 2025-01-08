.. Copyright 2019 Intel Corporation
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

.. re-use for math equations:
.. |x_vector| replace:: :math:`(x_1, \ldots, x_p)`

.. _linear_regression:

Linear Regression
=================


.. toctree::
   :glob:
   :maxdepth: 4

Linear regression is a method for modeling the relationship between a
dependent variable (which may be a vector) and one or more
explanatory variables by fitting linear equations to observed data.

Details
*******

Let |x_vector| be a vector of input variables and
:math:`y=(y_1, \ldots, y_k)` be the response. For each :math:`j=1, \ldots ,k`,
the linear regression model has the format [Hastie2009]_:

.. math::
  y_j = \beta_{0j} + \beta_{1j} x_1 + \ldots + \beta_{pj}  x_p


Here :math:`x_i`, :math:`i=1, \ldots,p`, are referred to as independent
variables, and :math:`y_j` are referred to as dependent variables
or responses.

Linear regression is called:

- **Simple Linear Regression** (if there is only one explanatory variable)
- **Multiple Linear Regression** (if the number of explanatory variables :math:`p > 1`)

Training Stage
--------------

Let :math:`(x_{11}, \ldots, x_{1p}, y_1, \ldots, x_{n1}, \ldots, x_{np}, y_n)` be a set of
training data, :math:`n \gg p`. The matrix :math:`X` of size :math:`n \times p` contains
observations :math:`x_{ij}`, :math:`i=1, \ldots, n`, :math:`j = 1, \ldots, p` of independent
variables.

To estimate the coefficients :math:`(\beta_{0j}, \ldots, \beta_{pj})`
one these methods can be used:

-  Normal Equation system

-  QR matrix decomposition

Prediction Stage
----------------

Linear regression based prediction is done for input vector |x_vector|
using the equation :math:`y_j = \beta_{0j} + \beta_{1j}x_1 + \ldots + \beta_{pj}x_p`
for each :math:`j=1, \ldots, k`.

Usage of Training Alternative
*****************************

To build a Linear Regression model using methods of the Model Builder class of Linear Regression, complete the following steps:

- Create a Linear Regression model builder using a constructor with the required number of responses and features.
- Use the ``setBeta`` method to add the set of pre-calculated coefficients to the model.
  Specify random access iterators to the first and the last element of the set of coefficients [ISO/IEC 14882:2011 §24.2.7]_.

  .. note::

    If your set of coefficients does not contain an intercept,
    ``interceptFlag`` is automatically set to ``False``, and to ``True``, otherwise.

- Use the ``getModel`` method to get the trained Linear Regression model.
- Use the ``getStatus`` method to check the status of the model building process.
  If ``DAAL_NOTHROW_EXCEPTIONS`` macros is defined, the status report contains the list of errors
  that describe the problems API encountered (in case of API runtime failure).

.. note::

  If after calling the ``getModel`` method you use the ``setBeta`` method to update coefficients,
  the initial model will be automatically updated with the new :math:`\beta` coefficients.

Examples
--------

.. tabs::

  .. tab:: C++ (CPU)

    - :cpp_example:`lin_reg_model_builder.cpp <linear_regression/lin_reg_model_builder.cpp>`

  .. tab:: Python*

    - :daal4py_example:`linear_regression.py`
