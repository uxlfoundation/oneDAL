.. Copyright contributors to the oneDAL project
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

.. _alg_hdbscan:

=======
HDBSCAN
=======

.. include:: ../../../includes/clustering/hdbscan-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _hdbscan_c_math:

Computation
-----------

Given the set :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})\}`
of :math:`n` :math:`p`-dimensional feature vectors, a ``metric`` for pairwise distance computation,
and positive integers ``min_cluster_size`` and ``min_samples``, the problem is to compute cluster
assignments for each observation.

.. _hdbscan_c_math_metrics:

Supported distance metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following distance metrics are supported via the ``metric`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 20 40 20

   * - Metric
     - Formula
     - Methods
   * - Euclidean (default)
     - :math:`d(x, y) = \sqrt{\sum_i (x_i - y_i)^2}`
     - brute_force, kd_tree
   * - Manhattan
     - :math:`d(x, y) = \sum_i |x_i - y_i|`
     - brute_force, kd_tree
   * - Minkowski
     - :math:`d(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{1/p}`
     - brute_force, kd_tree
   * - Chebyshev
     - :math:`d(x, y) = \max_i |x_i - y_i|`
     - brute_force, kd_tree
   * - Cosine
     - :math:`d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}`
     - brute_force only

The Minkowski metric requires a ``degree`` parameter :math:`p > 0`. Setting :math:`p = 1` is
equivalent to Manhattan, :math:`p = 2` is equivalent to Euclidean.

The Cosine metric is not compatible with the kd_tree method because no valid bounding-box
lower bound exists for cosine distance in a k-d tree.

HDBSCAN consists of the following steps:

**(1) Core distance computation:**
For each point :math:`x_i`, the core distance is the distance to its
:math:`k`-th nearest neighbor, where :math:`k` = ``min_samples``:

.. math::
   d_{\text{core}}(x_i) = d(x_i, \text{NN}_k(x_i))

**(2) Mutual Reachability Distance (MRD):**
The mutual reachability distance between any two points :math:`x_i` and :math:`x_j` is defined as:

.. math::
   d_{\text{mrd}}(x_i, x_j) = \max\{d_{\text{core}}(x_i),\; d_{\text{core}}(x_j),\; d(x_i, x_j)\}

This metric upweights sparse regions, making the algorithm robust to varying densities.

**(3) Minimum Spanning Tree (MST):**
Construct the minimum spanning tree of the complete graph where each observation
is a vertex and edge weights are mutual reachability distances.

**(4) Condensed cluster tree:**
Convert the MST into a dendrogram (hierarchical clustering), then condense it by
removing splits where one of the child clusters has fewer than ``min_cluster_size``
points. Points that fall out of clusters during condensation are potential noise points.

**(5) Cluster extraction (Excess of Mass):**
Using the Excess of Mass (EOM) method [Campello2013]_, compute the stability of each
cluster in the condensed tree, then select the set of clusters that maximizes total
stability. Selected clusters provide the flat clustering output.

Each cluster gets a unique identifier, an integer from :math:`0` to
:math:`\text{cluster\_count} - 1`. Observations not belonging to any cluster
are assigned :math:`-1` (noise).


.. _hdbscan_c_math_brute_force:

Computation method: *brute_force*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The brute-force method computes the full :math:`n \times n` pairwise distance matrix
using the selected metric, then uses this matrix for core distance computation
and Prim's MST algorithm. Euclidean and Cosine distances are GEMM-accelerated;
other metrics use element-wise computation.

- **Core distances**: For each row of the distance matrix, find the :math:`k`-th
  nearest distance using partial sort.
- **MST**: Prim's algorithm with :math:`O(n^2)` complexity, using precomputed MRD matrix.
- **Complexity**: :math:`O(n^2 p)` for distances + :math:`O(n^2)` for MST.
- **Memory**: :math:`O(n^2)` for the distance matrix.


.. _hdbscan_c_math_kd_tree:

Computation method: *kd_tree*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The k-d tree method avoids the :math:`O(n^2)` distance matrix by building a k-d tree
data structure over the input data and using it for both core distance computation
and MST construction via Boruvka's algorithm. Supports Euclidean, Manhattan,
Minkowski, and Chebyshev metrics (Cosine is not supported).

- **Core distances**: For each point, perform a k-NN query on the k-d tree.
  Complexity: :math:`O(n \cdot k \cdot \log n)`.
- **MST**: Boruvka's algorithm with k-d tree-accelerated nearest-different-component
  queries, using three-level pruning (component, bounding-box MRD lower bound,
  near/far child ordering). Complexity: :math:`O(n \log^2 n)`.
- **Memory**: :math:`O(n \cdot p)` for k-d tree nodes and bounding boxes.

The k-d tree method is significantly faster for moderate-sized datasets,
especially in low-to-moderate dimensions.


---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: HDBSCAN <api_hdbscan>`.

--------
Examples
--------

.. include:: ../../../includes/clustering/hdbscan-examples.rst

----------
References
----------

.. [Campello2013] Campello, R.J.G.B., Moulavi, D., Sander, J.
   *Density-Based Clustering Based on Hierarchical Density Estimates*.
   Advances in Knowledge Discovery and Data Mining, 2013.
