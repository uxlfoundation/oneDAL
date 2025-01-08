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

.. _alg_knn:

=================================================================
k-Nearest Neighbors Classification, Regression, and Search (k-NN)
=================================================================

.. include:: ../../../includes/nearest-neighbors/knn-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _knn_t_math:

Training
--------
.. tabs::

  .. group-tab:: Classification

   Let :math:`X = \{ x_1, \ldots, x_n \}` be the training set of
   :math:`p`-dimensional feature vectors, let :math:`Y = \{ y_1, \ldots, y_n \}` be
   the set of class labels, where :math:`y_i \in \{ 0, \ldots, C-1 \}`, :math:`1
   \leq i \leq n`, and :math:`C` is the number of classes. Given :math:`X`,
   :math:`Y`, and the number of nearest neighbors :math:`k`,
   the problem is to build a model that allows distance computation
   between the feature vectors in training and inference sets at the inference
   stage.

  .. group-tab:: Regression

   Let :math:`X = \{ x_1, \ldots, x_n \}` be the training set of
   :math:`p`-dimensional feature vectors, let :math:`Y = \{ y_1, \ldots, y_n \}` be
   the corresponding continuous target outputs, where :math:`y_i \in \mathbb{R}`. Given :math:`X`,
   :math:`Y`, and the number of nearest neighbors :math:`k`,
   the problem is to build a model that allows distance computation
   between the feature vectors in training and inference sets at the inference
   stage.

  .. group-tab:: Search

   Let :math:`X = \{ x_1, \ldots, x_n \}` be the training set of
   :math:`p`-dimensional feature vectors. Given :math:`X` and the number of nearest neighbors
   :math:`k`, the problem is to build a model that allows distance computation
   between the feature vectors in training and inference sets at the inference
   stage.

.. _knn_t_math_brute_force:

Training method: *brute-force*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training operation produces the model that stores all the feature vectors
from the initial training set :math:`X`.

.. _knn_t_math_kd_tree:

Training method: *k-d tree*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training operation builds a :math:`k`-:math:`d` tree that partitions the
training set :math:`X` (for more details, see :txtref:`k-d Tree <kd_tree>`).

.. _knn_i_math:

Inference
---------
.. tabs::

  .. group-tab:: Classification

   Let :math:`X' = \{ x_1', \ldots, x_m' \}` be the inference set of
   :math:`p`-dimensional feature vectors. Given :math:`X'`, the model produced at
   the training stage, and the number of nearest neighbors :math:`k`, the problem is
   to predict the label :math:`y_j'` from the :math:`Y` set for each :math:`x_j'`,
   :math:`1 \leq j \leq m`, by performing the following steps:

   #. Identify the set :math:`N(x_j') \subseteq X` of :math:`k` feature vectors
      in the training set that are nearest to :math:`x_j'` with respect to the
      Euclidean distance, which is chosen by default. The distance can be customized
      with the predefined set of pairwise distances: :ref:`Minkowski distances
      <alg_minkowski_distance>` with fractional degree (including Euclidean distance),
      :ref:`Chebyshev distance <alg_chebyshev_distance>`, and
      :ref:`Cosine distance <alg_cosine_distance>`.

   #. Estimate the conditional probability for the :math:`l`-th class as the
      fraction of vectors in :math:`N(x_j')` whose labels :math:`y_j` are equal to
      :math:`l`:

      .. math::
         :label: p_predict

         P_{jl} = \frac{1}{| N(x_j') |} \Big| \big\{ x_r \in N(x_j') : y_r = l
         \big\} \Big|, \quad 1 \leq j \leq m, \; 0 \leq l < C.


   #. Predict the class that has the highest probability for the feature vector
      :math:`x_j'`:

      .. math::
         :label: y_predict

         y_j' = \mathrm{arg}\max_{0 \leq l < C} P_{jl},
         \quad 1 \leq j \leq m.

  .. group-tab:: Regression

   Let :math:`X' = \{ x_1', \ldots, x_m' \}` be the inference set of
   :math:`p`-dimensional feature vectors. Given :math:`X'`, the model produced at
   the training stage, and the number of nearest neighbors :math:`k`, the problem is
   to predict the continuous target variable :math:`y_j'` from the :math:`Y` set for each :math:`x_j'`,
   :math:`1 \leq j \leq m`, by performing the following steps:

   #. Identify the set :math:`N(x_j') \subseteq X` of :math:`k` feature vectors
      in the training set that are nearest to :math:`x_j'` with respect to the
      Euclidean distance, which is chosen by default. The distance can be customized
      with the predefined set of pairwise distances: :ref:`Minkowski distances
      <alg_minkowski_distance>` with fractional degree (including Euclidean distance),
      :ref:`Chebyshev distance <alg_chebyshev_distance>`, and
      :ref:`Cosine distance <alg_cosine_distance>`.

   #. Estimate the conditional expectation of the target variable based on the nearest neighbors
      :math:`N(x_j')` as the average of the target values for those neighbors:

      .. math::
         :label: y_predict_reg

         y_j' = \frac{1}{| N(x_j') |} \sum_{x_r \in N(x_j')} y_r,
         \quad 1 \leq j \leq m.

  .. group-tab:: Search

   Let :math:`X' = \{ x_1', \ldots, x_m' \}` be the inference set of
   :math:`p`-dimensional feature vectors. Given :math:`X'`, the model produced at
   the training stage, and the number of nearest neighbors :math:`k`:

   #. Identify the set :math:`N(x_j') \subseteq X` of :math:`k` feature vectors
      in the training set that are nearest to :math:`x_j'` with respect to the
      Euclidean distance, which is chosen by default. The distance can be customized
      with the predefined set of pairwise distances: :ref:`Minkowski distances
      <alg_minkowski_distance>` with fractional degree (including Euclidean distance),
      :ref:`Chebyshev distance <alg_chebyshev_distance>`, and
      :ref:`Cosine distance <alg_cosine_distance>`.

.. _knn_i_math_brute_force:

Inference method: *brute-force*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Brute-force inference method determines the set :math:`N(x_j')` of the
nearest feature vectors by iterating over all the pairs :math:`(x_j', x_i)` in
the implementation defined order, :math:`1 \leq i \leq n`, :math:`1 \leq j \leq
m`.

.. _knn_i_math_kd_tree:

Inference method: *k-d tree*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
K-d tree inference method traverses the :math:`k`-:math:`d` tree to find feature
vectors associated with a leaf node that are closest to :math:`x_j'`, :math:`1
\leq j \leq m`. The set :math:`\tilde{n}(x_j')` of the currently known nearest
:math:`k` neighbors is progressively updated during the tree traversal. The
search algorithm limits exploration of the nodes for which the distance between
the :math:`x_j'` and respective part of the feature space is not less than the
distance between :math:`x_j'` and the most distant feature vector from
:math:`\tilde{n}(x_j')`. Once tree traversal is finished, :math:`\tilde{n}(x_j')
\equiv N(x_j')`.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: k-Nearest Neighbors Classification, Regression, and Search <api_knn>`.

The following table describes current device support:

+----------------+------+------+
| Task           | CPU  | GPU  |
+================+======+======+
| Classification | Yes  | Yes  |
+----------------+------+------+
| Regression     | No   | Yes  |
+----------------+------+------+
| Search         | Yes  | Yes  |
+----------------+------+------+

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SPMD mode (only on GPU).

-------------
Usage Example
-------------

.. include:: ../../../includes/nearest-neighbors/knn-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/nearest-neighbors/knn-examples.rst
