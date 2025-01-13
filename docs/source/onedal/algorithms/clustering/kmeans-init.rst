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

.. highlight:: cpp
.. default-domain:: cpp

.. _alg_kmeans_init:

======================
K-Means initialization
======================

.. include:: ../../../includes/clustering/kmeans-init-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _kmeans_init_c_math:

Computing
---------

Given the training set :math:`X = \{ x_1, \ldots, x_n \}` of
:math:`p`-dimensional feature vectors and a positive integer :math:`k`, the
problem is to find a set :math:`C = \{ c_1, \ldots, c_k \}` of
:math:`p`-dimensional initial centroids.

.. _kmeans_init_c_math_dense:

Computing method: *dense*
-------------------------

The method chooses first :math:`k` feature vectors from the training set
:math:`X`.

.. _kmeans_init_c_math_random_dense:

Computing method: *random_dense*
--------------------------------

The method chooses random :math:`k` feature vectors from the training set
:math:`X`.

.. _kmeans_init_c_math_plus_plus_dense:

Computing method: *plus_plus_dense* (only on CPU)
-------------------------------------------------

The method is designed as follows: the first centroid :math:`c_0` is selected randomly and :math:`C = \{ c_1 \}`.
Then the following step is repeated until :math:`C` reaches the necessary size.

.. math::
   c_{i+1} = \mathrm{arg}\max_{x \in X} d^2(x, C), C = C \cup \{ c_{i+1} \}

.. _kmeans_init_c_math_parallel_plus_dense:

Computing method: *parallel_plus_dense* (only on CPU)
-----------------------------------------------------

The method is the same as :ref:`K-Means++ <kmeans_init_c_math_plus_plus_dense>`, but the data is divided into equal parts and the
algorithm runs on each of them.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: K-Means initialization <api_kmeans_init>`.

-------------
Usage Example
-------------

.. include:: ../../../includes/clustering/kmeans-init-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/clustering/kmeans-init-examples.rst
