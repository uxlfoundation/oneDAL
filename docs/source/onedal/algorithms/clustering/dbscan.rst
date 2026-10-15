.. Copyright 2021 Intel Corporation
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

.. re-use for math equations:
.. |x| replace:: :math:`x`
.. |y| replace:: :math:`y`

.. _alg_dbscan:

======
DBSCAN
======

.. include:: ../../../includes/clustering/dbscan-introduction.rst

------------------------
Mathematical formulation
------------------------

.. _dbscan_c_math:

Computation
-----------
Given the set :math:`X = \{x_1 = (x_{11}, \ldots, x_{1p}), \ldots, x_n = (x_{n1}, \ldots, x_{np})\}`
of :math:`n` :math:`p`-dimensional feature vectors (further referred as observations),
a ``metric`` for pairwise distance computation, a positive floating-point number ``epsilon``
and a positive integer ``minObservations``,
the problem is to get clustering assignments for each input observation, based on the definitions below [Ester96]_:
two observations |x| and |y| are considered to be in the same cluster if there is a :term:`core observation` :math:`z`,
and |x| and |y| are both :term:`reachable` from :math:`z`.

Each cluster gets a unique identifier, an integer number from :math:`0` to :math:`\text{total number of clusters } – 1`.
Each observation is assigned an identifier of the :term:`cluster` it belongs to,
or :math:`-1` if the observation considered to be a :term:`noise observation`.

.. _dbscan_c_math_metrics:

Supported distance metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

The distance used to decide whether |y| lies in the ``epsilon``-neighborhood of
|x| is selected via the ``metric`` parameter. DBSCAN reuses the
:ref:`Minkowski distances <alg_minkowski_distance>` family with configurable
``degree`` :math:`p > 0`: ``degree = 1`` yields the Manhattan (:math:`L_1`)
distance, ``degree = 2`` yields the Euclidean (:math:`L_2`, default) distance,
and the limit :math:`p \to \infty` corresponds to the
:ref:`Chebyshev distance <alg_chebyshev_distance>` (:math:`L_\infty`). The
Manhattan, Euclidean and Chebyshev distances are also exposed as their own
``metric`` values. In addition, DBSCAN supports the
:ref:`Cosine distance <alg_cosine_distance>` for angular data. All metrics are
supported on CPU and GPU, in both memory modes.

``epsilon`` is interpreted in the units of the selected metric, so it generally
has to be re-tuned after changing the metric. As in scikit-learn, an all-zero
observation is at Cosine distance :math:`1` from every other observation.

---------------------
Programming Interface
---------------------

Refer to :ref:`API Reference: DBSCAN <api_dbscan>`.

----------------
Distributed mode
----------------

The algorithm supports distributed execution in SPMD mode (only on GPU).

-------------
Usage Example
-------------

.. include:: ../../../includes/clustering/dbscan-usage-examples.rst

--------
Examples
--------

.. include:: ../../../includes/clustering/dbscan-examples.rst
