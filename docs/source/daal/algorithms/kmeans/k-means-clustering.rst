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

.. _kmeans:

K-Means Clustering
==================

.. note::

   K-Means and K-Means initialization are also available with oneAPI interfaces:

   - :ref:`alg_kmeans`
   - :ref:`alg_kmeans_init`

.. toctree::
   :glob:
   :maxdepth: 4



K-Means is among the most popular and simplest clustering methods. It
is intended to partition a data set into a small number of clusters
such that feature vectors within a cluster have greater similarity
with one another than with feature vectors from other clusters. Each
cluster is characterized by a representative point, called a
centroid, and a cluster radius.

In other words, the clustering methods enable reducing the problem of
analysis of the entire data set to the analysis of clusters.

There are numerous ways to define the measure of similarity and
centroids. For K-Means, the centroid is defined as the mean of
feature vectors within the cluster.

.. _kmeans_details:

Details
*******

Given the set :math:`X = \{ x_1= (x_{11},\ldots{},x_{1p} ), \ldots{}, x_n = (x_{n1},\ldots{},x_{np} ) \}`
of :math:`n` :math:`p`-dimensional feature vectors and a positive integer :math:`k`, the
problem is to find a set :math:`C = \{ c_1, \ldots{} , c_k \}`
of :math:`k` :math:`p`-dimensional vectors that minimize the objective function
(overall error)

.. math::

	{\text{Φ}}_{X}(C)=\sum _{{x}_{i\in X}}{d}^{2}({x}_{i},C)


where :math:`{d}^{2}({x}_{i},C)` is the distance from :math:`x_i`
to the closest center in :math:`C`, such as the Euclidean distance.
The vectors :math:`c_1, \ldots{} , c_k` are called centroids. To start
computations, the algorithm requires initial values of centroids.

Centroid Initialization
-----------------------

Centroids initialization can be done using these methods:

- Choice of first :math:`k` feature vectors from the data set :math:`X`.

-
 Random choice of :math:`k` feature vectors from the data set using the
 following simple random sampling draw-by-draw algorithm. The
 algorithm does the following:

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with
	equal probability.

 #.
	Excludes :math:`x_i` from :math:`X` and adds it to the current set of
	centers.

 #.
	Resumes from step 1 until the set of centers
	reaches the desired size :math:`k`.

-
 K-Means++ algorithm [Arthur2007]_,
 which selects centers with the probability proportional to
 their contribution to the overall error :math:`{\text{Φ}}_{X}(C)`
 according to the following scheme:

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with
	equal probability.

 #.
	Excludes :math:`x_i` from :math:`X` and adds it to the current set of
	centers :math:`C`.

 #.
	For each feature vector :math:`x_i` in :math:`X` calculates its
	minimal distance :math:`{d}({x}_{i},C)` from the current set of
	centers :math:`C`.

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with the probability :math:`\frac{{d}^{2}({x}_{i},C)}{{\text{ Φ}}_{X}(C)}`.

 #.
	Resumes from step 2 until the set of centers
	:math:`C` reaches the desired size :math:`k`.

-
 Parallel K-Means++ algorithm [Bahmani2012]_
 that does the following:

 #.
	Chooses one of the feature vectors :math:`x_i` from :math:`X` with
	equal probability.

 #.
	Excludes :math:`x_i` from :math:`X` and adds it to the current set of
	centers :math:`C`.

 #.
	Repeats :math:`nRounds` times:

	a.
	   For each feature vector :math:`x_i` from :math:`X` calculates its
	   minimal distance :math:`{d}({x}_{i},C)` from the current set
	   of centers :math:`C`.

	b.
	   Chooses :math:`L = oversamplingFactor \cdot k` feature vectors :math:`x_i` from :math:`X` with the probability :math:`\frac{{d}^{2}({x}_{i},C)}{{\text{ Φ}}_{X}(C)}`.

	c.
	   Excludes :math:`x_i` vectors chosen in the previous step
	   from :math:`X` and adds them to the current set of centers :math:`C`.

 #.
	For :math:`c_i \in C` sets :math:`w_i` to the ratings, the
	number of points in :math:`X` closer to :math:`c_i` than to any other
	point in :math:`C`.

 #.
	Applies K-Means++ algorithm with weights :math:`w_i` to the
	points in :math:`C`, which means that the following probability is
	used in step:

	.. math::
	   \frac{{w}_{i}{d}^{2}({x}_{i},C)}{\text{ }{\sum }_{{x}_{iϵX}}{w}_{i}{d}^{2}({x}_{i},C)}


 The algorithm parameters define the number of candidates :math:`L`
 selected in each round and number of rounds:

 -

	Choose :math:`oversamplingFactor` to make :math:`L = O(k)`.

 -

	Choose nRounds as :math:`O(\log({\text{Φ}}_{X}(C)))`, where :math:`{\text{Φ}}_{X}(C)`
	is the estimation of the goal function when the first center
	is chosen.
	[Bahmani2012]_ recommends to set :math:`nRounds` to a constant value not greater
	than :math:`8`.


Computation
-----------

Computation of the goal function includes computation of the
Euclidean distance between vectors :math:`||x_j - m_i||`.
The algorithm uses the following modification of the Euclidean
distance between feature vectors :math:`a` and :math:`b`: :math:`d(a,b) = d_1(a,b) + d_2(a,b)`, where :math:`d_1` is computed
for continuous features as

.. math::
 {d}_{1}(a,b)=\sqrt{{\sum }_{k=1}^{{p}_{1}}{({a}_{k}-{b}_{k})}^{2}}

and :math:`d_2` is computed for binary categorical features as

.. math::
 {d}_{2}(a,b)=\gamma \sqrt{\sum _{k=1}^{{p}_{2}}{({a}_{k}-{b}_{k})}^{2}}

In these equations, :math:`\gamma` γ weighs the impact of binary categorical
features on the clustering, :math:`p_1` is the number of continuous
features, and :math:`p_2` is the number of binary categorical
features. Note that the algorithm does not support non-binary
categorical features.

The K-Means clustering algorithm computes centroids using Lloyd's
method [Lloyd82]_.
For each feature vector :math:`x_1, \ldots{} , x_k`, you can also
compute the index of the cluster that contains the feature vector.

In some cases, if no vectors are assigned to some clusters on a
particular iteration, the iteration produces an empty cluster. It
may occur due to bad initialization of centroids or the dataset
structure. In this case, the algorithm uses the following strategy
to replace the empty cluster centers and decrease the value of the
overall goal function:

-

 Feature vectors, most distant from their assigned centroids,
 are selected as the new cluster centers. Information about
 these vectors is gathered automatically during the algorithm
 execution.

-

 In the distributed processing mode, most distant vectors from
 the local nodes are computed (Step 1), stored in
 *PartialResult*, and collected on the master node (Step 2). For
 more details, see the *PartialResult* description at Step 1 [Tan2005]_.

.. _kmeans_init:

Initialization
**************

The K-Means clustering algorithm requires initialization of
centroids as an explicit step. Initialization flow depends on the
computation mode. Skip this step if you already calculated initial
centroids.

For initialization, the following computation modes are available:

.. toctree::
   :maxdepth: 1

   initialization-batch.rst
   initialization-distributed.rst

Computation
***********

The following computation modes are available:

.. toctree::
   :maxdepth: 1

   computation-batch.rst
   computation-distributed.rst

.. note:: Distributed mode is not available for oneAPI interfaces and for Python* with DPC++ support.

Examples
********

.. tabs::

  .. tab:: oneAPI DPC++

    Batch Processing:

    - :ref:`dpc_kmeans_init_dense.cpp`
    - :ref:`dpc_kmeans_lloyd_dense_batch.cpp`

  .. tab:: oneAPI C++

    Batch Processing:

    - :ref:`cpp_kmeans_lloyd_dense_batch.cpp`
    - :ref:`cpp_kmeans_init_dense.cpp`

  .. tab:: C++ (CPU)

    Batch Processing:

    - :cpp_example:`kmeans_dense_batch.cpp <kmeans/kmeans_dense_batch.cpp>`
    - :cpp_example:`kmeans_csr_batch.cpp <kmeans/kmeans_csr_batch.cpp>`

    Distributed Processing:

    - :cpp_example:`kmeans_dense_distr.cpp <kmeans/kmeans_dense_distr.cpp>`
    - :cpp_example:`kmeans_csr_distr.cpp <kmeans/kmeans_csr_distr.cpp>`

  .. tab:: Python*

    Batch Processing:

    - :daal4py_example:`kmeans.py`

    Distributed Processing

    - :daal4py_example:`kmeans_spmd.py`

Performance Considerations
**************************

To get the best overall performance of the K-Means algorithm:

-
  If input data is homogeneous, provide the input data and store
  results in homogeneous numeric tables of the same type as
  specified in the algorithmFPType class template parameter.

-
  If input data is non-homogeneous, use AOS layout rather than SOA
  layout.

-
  For the output assignments table, use a homogeneous numeric table
  of the int type.



.. include:: ../../../opt-notice.rst
