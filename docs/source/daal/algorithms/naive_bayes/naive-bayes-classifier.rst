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

Naïve Bayes Classifier
======================

Naïve Bayes is a set of simple and powerful classification methods
often used for text classification, medical diagnosis, and other
classification problems. In spite of their main assumption about
independence between features, Naïve Bayes classifiers often work
well when this assumption does not hold. An advantage of this method
is that it requires only a small amount of training data to estimate
model parameters.


.. toctree::
   :glob:
   :maxdepth: 4


Details
*******

The library provides Multinomial Naïve Bayes classifier [Renie03]_.

Let :math:`J` be the number of classes, indexed :math:`0, 1, \ldots, J-1`. The integer-valued
feature vector :math:`x_i = (x_{11}, \ldots, x_{ip})`, :math:`i=1, \ldots, n`,
contains scaled frequencies: the value of :math:`x_{ik}` is the number
of times the :math:`k`-th feature is observed in the vector :math:`x_i` (in
terms of the document classification problem, :math:`x_{ik}` is the
number of occurrences of the word indexed :math:`k` in the document :math:`x_i`.
For a given data set (a set of :math:`n` documents), :math:`(x_1, \ldots, x_n)`, the problem is to train a Naïve Bayes
classifier.

Training Stage
--------------

The Training stage involves calculation of these parameters:

- :math:`\mathrm{log}\left({\theta }_{jk}\right)=\mathrm{log}\left(\frac{{N}_{jk}+{\alpha }_{k}}{{N}_{j}+\alpha }\right)`, where
  :math:`N_{jk}` is the number of occurrences of the feature :math:`k` in the class :math:`j`,
  :math:`N_j` is the total number of occurrences of all features in the class,
  the :math:`\alpha_k`the parameter is the imagined number of occurrences of the feature :math:`k`
  (for example, :math:`\alpha_k = 1`), and :math:`\alpha` is the sum of all :math:`\alpha_k`.

- :math:`\mathrm{log}\left({\theta }_{j}\right)`, where :math:`p(\theta_j)` is the prior class estimate.

Prediction Stage
----------------

Given a new feature vector :math:`x_i`, the classifier determines the class the vector belongs to:

.. math::
	class\left({x}_{i}\right)=\mathrm{arg}{\mathrm{max}}_{j}\left(\mathrm{log}\left(p\left({\theta }_{j}\right)\right)+{\sum }_{k}\mathrm{log}\left({\theta }_{jk}\right)\right).

Computation
***********

The following computation modes are available:

.. toctree::
   :maxdepth: 1

   computation-batch.rst
   computation-online.rst
   computation-distributed.rst

Examples
********

.. tabs::

   .. tab:: C++ (CPU)

      Batch Processing:

      -  :cpp_example:`mn_naive_bayes_dense_batch.cpp <naive_bayes/mn_naive_bayes_dense_batch.cpp>`
      -  :cpp_example:`mn_naive_bayes_csr_batch.cpp <naive_bayes/mn_naive_bayes_csr_batch.cpp>`

      Online Processing:

      - :cpp_example:`mn_naive_bayes_dense_online.cpp <naive_bayes/mn_naive_bayes_dense_online.cpp>`
      - :cpp_example:`mn_naive_bayes_csr_online.cpp <naive_bayes/mn_naive_bayes_csr_online.cpp>`

      Distributed Processing:

      - :cpp_example:`mn_naive_bayes_dense_distr.cpp <naive_bayes/mn_naive_bayes_dense_distr.cpp>`
      - :cpp_example:`mn_naive_bayes_csr_distr.cpp <naive_bayes/mn_naive_bayes_csr_distr.cpp>`

   .. tab:: Python*

      Batch Processing:

      - :daal4py_example:`naive_bayes.py`

      Online Processing:

      - :daal4py_example:`naive_bayes_streaming.py`

      Distributed Processing:

      - :daal4py_example:`naive_bayes_spmd.py`

Performance Considerations
**************************

Training Stage
--------------


To get the best overall performance at the Naïve Bayes classifier
training stage:

- If input data is homogeneous:

  -  For the training data set, use a homogeneous numeric table
     of the same type as specified in the algorithmFPType class
     template parameter.
  -  For class labels, use a homogeneous numeric table of type int.

- If input data is non-homogeneous, use AOS layout rather than SOA layout.

The training stage of the Naïve Bayes classifier algorithm is
memory access bound in most cases. Therefore, use efficient data
layout whenever possible.

Prediction Stage
----------------

To get the best overall performance at the Naïve Bayes classifier
prediction stage:

-  For the working data set, use a homogeneous numeric table of
   the same type as specified in the algorithmFPType class
   template parameter.
-  For predicted labels, use a homogeneous numeric table of type int.


.. include:: ../../../opt-notice.rst